"""FedSAA: Similarity-Attended Aggregation.

Per round, per LoRA target layer, given each participating client's (A_i, B_i):
  1. U_i = B_i @ A_i, flattened.
  2. S_ij = cosine(U_i, U_j)                          -- similarity matrix (saved for figure F4).
  3. w_ij = softmax_j(S_ij / tau).
  4. Ubar_i = sum_j w_ij U_j ; G = mean_j U_j.
  5. Utilde_i = (1 - lambda) * Ubar_i + lambda * G.
  6. truncated rank-r SVD of Utilde_i -> per-client (A_i, B_i), returned individually.

Unlike FedAvg, the result is PERSONALIZED: every participating client gets back its own
(A_i, B_i), not one shared global adapter. lambda=1 collapses Utilde_i to G for every client
(independent of tau/w_ij), which is exactly "FedAvg on merged LoRA deltas" -- see
tests/test_fedsaa.py for the equivalence check.

Local adapter retention (step 7): the server-computed personalized adapter is then blended with
each client's OWN pre-aggregation local adapter (the one it just sent, before this round's
attention/SVD step touched it) via a retention coefficient alpha:

    final_i = (1 - alpha) * global_personalized_i + alpha * local_i

alpha=0 is pure FedSAA as before (fully replace with the server's result); alpha=1 would ignore
the server's aggregation entirely and keep the client's own just-trained adapter. Applied per
LoRA tensor (A and B separately, same shapes, simple elementwise blend).

Post-aggregation local adaptation (this step): after retention-blending, each participating
client runs a brief additional local fine-tuning pass (`local_adapt_steps`, a handful of SGD
mini-batches, not a full epoch) on its own data before evaluation -- otherwise "personalization"
would only ever happen at the server and the client's adapter would never actually see its own
data again after being overwritten by the blended result.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from src.data.client_data import build_clients, build_global_test_loader
from src.federated.client import local_adapt_steps, local_train
from src.federated.common import log_cka_this_round, rss_mb, run_dir, write_metrics_csv
from src.metrics.cka import build_probe_batch
from src.metrics.evaluation import evaluate_client, evaluate_global
from src.models.client_model import build_client_model
from src.models.lora_model import adapter_payload_mb, get_adapter_state, set_adapter_state

LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


def _layer_names(adapter_state: dict) -> list[str]:
    return sorted(k[: -len(LORA_A_SUFFIX)] for k in adapter_state if k.endswith(LORA_A_SUFFIX))


def fedsaa_aggregate(adapter_states: list[dict], tau: float, lam: float, svd_rank: int,
                      alpha: float = 0.0):
    """Returns (per_client_states, similarity_by_layer).

    per_client_states[i] is client i's new personalized (A, B) adapter state dict, after local
    adapter retention: final_i = (1 - alpha) * global_personalized_i + alpha * local_i, where
    local_i = adapter_states[i] is that same client's own pre-aggregation adapter (the one it
    sent into this call, before steps 1-6 below touched it). alpha=0 (default) is pure FedSAA
    with no retention -- callers that want retention must pass alpha explicitly.
    similarity_by_layer[layer_name] is the (n_clients, n_clients) cosine similarity matrix S.
    """
    n = len(adapter_states)
    layers = _layer_names(adapter_states[0])

    per_client_states = [dict() for _ in range(n)]
    similarity_by_layer: dict[str, np.ndarray] = {}

    for layer in layers:
        a_key, b_key = layer + LORA_A_SUFFIX, layer + LORA_B_SUFFIX
        a_shape = adapter_states[0][a_key].shape  # (r, in_ch, kh, kw)
        b_shape = adapter_states[0][b_key].shape  # (out_ch, r, 1, 1)
        r = a_shape[0]
        out_ch = b_shape[0]
        in_flat = int(np.prod(a_shape[1:]))

        U_list = []
        for state in adapter_states:
            A = state[a_key].reshape(r, in_flat)
            B = state[b_key].reshape(out_ch, r)
            U_list.append(B @ A)  # (out_ch, in_flat)
        U_stack = torch.stack(U_list, dim=0)          # (n, out_ch, in_flat)
        U_flat = U_stack.reshape(n, -1)                # (n, D)

        # 2-3. cosine similarity + softmax attention weights
        U_norm = F.normalize(U_flat, dim=1, eps=1e-12)
        S = U_norm @ U_norm.T                          # (n, n)
        weights = F.softmax(S / tau, dim=1)            # softmax_j(S_ij / tau), rows sum to 1
        similarity_by_layer[layer] = S.numpy()

        # 4-5. attended aggregate blended with the uniform global mean
        Ubar = weights @ U_flat                         # (n, D)
        G = U_flat.mean(dim=0, keepdim=True)             # (1, D)
        Utilde = (1.0 - lam) * Ubar + lam * G             # (n, D), broadcasts G

        # 6. per-client truncated-SVD re-factorization back to rank svd_rank
        for i in range(n):
            Utilde_i = Utilde[i].reshape(out_ch, in_flat)
            U_svd, S_svd, Vt_svd = torch.linalg.svd(Utilde_i, full_matrices=False)
            k = min(svd_rank, S_svd.shape[0])
            sqrt_s = torch.sqrt(S_svd[:k].clamp(min=1e-12))
            B_new = U_svd[:, :k] * sqrt_s.unsqueeze(0)     # (out_ch, k)
            A_new = sqrt_s.unsqueeze(1) * Vt_svd[:k, :]     # (k, in_flat)

            if k < r:  # pad up to the configured rank if SVD returned fewer components
                B_new = F.pad(B_new, (0, r - k))
                A_new = F.pad(A_new, (0, 0, 0, r - k))

            A_global = A_new.reshape(a_shape)
            B_global = B_new.reshape(b_shape)

            if alpha > 0.0:
                # local adapter retention: blend the server's personalized result back toward
                # this client's own pre-aggregation adapter, rather than fully overwriting it
                A_local = adapter_states[i][a_key]
                B_local = adapter_states[i][b_key]
                A_final = (1.0 - alpha) * A_global + alpha * A_local
                B_final = (1.0 - alpha) * B_global + alpha * B_local
            else:
                A_final, B_final = A_global, B_global

            per_client_states[i][a_key] = A_final.contiguous()
            per_client_states[i][b_key] = B_final.contiguous()

    return per_client_states, similarity_by_layer


def run_fedsaa(cfg: dict, seed: int):
    device = torch.device(cfg["device"])
    fed_cfg = cfg["federated"]
    saa_cfg = cfg["method"]["fedsaa"]
    tau, lam, svd_rank = saa_cfg["tau"], saa_cfg["lambda"], saa_cfg["svd_rank"]
    alpha = saa_cfg["alpha"]                          # local adapter retention coefficient
    post_agg_adapt_steps = saa_cfg["post_agg_adapt_steps"]  # brief local fine-tune after aggregation

    clients_data = build_clients(cfg, seed=seed)
    global_test_loader = build_global_test_loader(clients_data, cfg)
    client_models = [
        build_client_model(cfg, num_classes=cd.num_classes).to(device) for cd in clients_data
    ]

    # start every client from the same initial adapter, for a fair comparison with FedAvg/FedProx
    init_state = get_adapter_state(client_models[0])
    for m in client_models:
        set_adapter_state(m, init_state)
    personalized_states = [init_state for _ in client_models]
    payload_mb = adapter_payload_mb(init_state)

    out_dir = run_dir(cfg, "fedsaa", seed)
    csv_path = os.path.join(out_dir, "metrics.csv")
    sim_dir = os.path.join(out_dir, "similarity")
    cka_dir = os.path.join(out_dir, "cka")
    os.makedirs(sim_dir, exist_ok=True)
    probe_x = build_probe_batch(clients_data, n_per_client=4, seed=seed)
    print(f"Writing metrics to {csv_path}")
    print(f"Writing per-round similarity matrices (figure F4) to {sim_dir}")
    print(f"Writing per-round client-CKA matrices (figure F7) to {cka_dir}")

    rng = np.random.RandomState(seed)
    num_clients = fed_cfg["num_clients"]
    num_sampled = max(1, int(round(fed_cfg["client_fraction"] * num_clients)))

    cumulative_mb = 0.0
    rows = []

    for round_idx in range(1, fed_cfg["rounds"] + 1):
        t0 = time.time()

        participant_ids = sorted(rng.choice(num_clients, size=num_sampled, replace=False).tolist())

        collected_states = []
        for cid in participant_ids:
            set_adapter_state(client_models[cid], personalized_states[cid])
            adapter_state, _n_samples, _loss = local_train(client_models[cid],
                                                             clients_data[cid].train_loader, cfg, device)
            collected_states.append(adapter_state)

        new_personalized, similarity_by_layer = fedsaa_aggregate(
            collected_states, tau, lam, svd_rank, alpha=alpha,
        )
        for local_idx, cid in enumerate(participant_ids):
            personalized_states[cid] = new_personalized[local_idx]
            set_adapter_state(client_models[cid], personalized_states[cid])

            # brief post-aggregation local adaptation: personalization is the point, so each
            # client should actually fine-tune on its own data again after retention-blending,
            # not just receive a server-computed adapter it never locally adapts further
            local_adapt_steps(client_models[cid], clients_data[cid].train_loader, cfg, device,
                               num_steps=post_agg_adapt_steps)
            personalized_states[cid] = get_adapter_state(client_models[cid])

        np.savez(os.path.join(sim_dir, f"round_{round_idx:03d}.npz"), **similarity_by_layer)

        per_client_acc, macro_f1s, aucs = [], [], []
        for cid, cd in enumerate(clients_data):
            res = evaluate_client(client_models[cid], cd.test_loader, device)
            per_client_acc.append(res["acc"])
            macro_f1s.append(res["macro_f1"])
            aucs.append(res["auc"])

        global_res = evaluate_global(client_models, global_test_loader, device)
        mean_cka = log_cka_this_round(client_models, probe_x, device, cka_dir, round_idx)

        adapter_mb_round = num_sampled * payload_mb * 2
        cumulative_mb += adapter_mb_round
        wall_clock = time.time() - t0

        row = {
            "round": round_idx,
            "global_acc": global_res["acc"],
            "per_client_acc": ";".join(f"{a:.4f}" for a in per_client_acc),
            "macro_f1": np.nanmean(macro_f1s),
            "auc": np.nanmean(aucs),
            "adapter_MB_round": adapter_mb_round,
            "cumulative_MB": cumulative_mb,
            "wall_clock_sec": wall_clock,
            "ram_mb": rss_mb(),
        }
        rows.append(row)
        print(
            f"[round {round_idx}/{fed_cfg['rounds']}] global_acc={row['global_acc']:.4f} "
            f"macro_f1={row['macro_f1']:.4f} auc={row['auc']:.4f} "
            f"comm={adapter_mb_round:.3f}MB (cum={cumulative_mb:.3f}MB) "
            f"time={wall_clock:.1f}s ram={row['ram_mb']:.1f}MB mean_cka={mean_cka:.4f}"
        )

    write_metrics_csv(csv_path, rows)
    return csv_path, rows
