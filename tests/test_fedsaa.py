"""Unit tests for src/federated/fedsaa.py (no pytest dependency -- plain asserts).

Usage:
    python -m tests.test_fedsaa
"""

from __future__ import annotations

import torch

from src.federated.fedsaa import fedsaa_aggregate


def _random_adapter_state(n_clients: int, r: int, out_ch: int, in_ch: int, k: int, seed: int):
    """Builds n_clients synthetic (A, B) states for one Conv2d LoRA layer, shaped like peft's
    Conv2d LoRA: lora_A (r, in_ch, k, k), lora_B (out_ch, r, 1, 1)."""
    gen = torch.Generator().manual_seed(seed)
    states = []
    for _ in range(n_clients):
        A = torch.randn(r, in_ch, k, k, generator=gen)
        B = torch.randn(out_ch, r, 1, 1, generator=gen)
        states.append({
            "layer0.lora_A.default.weight": A,
            "layer0.lora_B.default.weight": B,
        })
    return states


def test_lambda1_matches_fedavg_on_merged_deltas():
    """With lambda=1, Utilde_i collapses to G = mean_j(U_j) for every client, regardless of
    tau/attention weights -- i.e. every client's reconstructed merged delta B_i@A_i should equal
    the *same* value: the plain average of every client's original B_j@A_j. This is exactly
    "FedAvg" in the sense the primer defines it (lambda=1, uniform aggregate)."""
    n_clients, r, out_ch, in_ch, k = 4, 4, 8, 6, 3
    states = _random_adapter_state(n_clients, r, out_ch, in_ch, k, seed=0)

    # ground truth: plain mean of the original merged deltas U_j = B_j @ A_j, truncated to rank r
    # (the mean of several rank-r matrices is generally higher rank, so the fair comparison is
    # against the *rank-r SVD truncation* of that mean -- which is what any rank-r LoRA baseline,
    # including FedSAA at lambda=1, is constrained to represent).
    U_list = []
    for s in states:
        A = s["layer0.lora_A.default.weight"].reshape(r, -1)
        B = s["layer0.lora_B.default.weight"].reshape(out_ch, r)
        U_list.append(B @ A)
    G_full = torch.stack(U_list).mean(dim=0)  # (out_ch, in_flat)
    U_svd, S_svd, Vt_svd = torch.linalg.svd(G_full, full_matrices=False)
    G_expected = (U_svd[:, :r] * S_svd[:r]) @ Vt_svd[:r, :]  # rank-r truncated SVD of the mean

    # tau is arbitrary here -- lambda=1 makes attention weights irrelevant to the result
    for tau in [0.01, 0.1, 1.0, 100.0]:
        new_states, similarity_by_layer = fedsaa_aggregate(states, tau=tau, lam=1.0, svd_rank=r)

        assert "layer0" in similarity_by_layer
        S = similarity_by_layer["layer0"]
        assert S.shape == (n_clients, n_clients)

        for i in range(n_clients):
            A_new = new_states[i]["layer0.lora_A.default.weight"].reshape(r, -1)
            B_new = new_states[i]["layer0.lora_B.default.weight"].reshape(out_ch, r)
            recon = B_new @ A_new
            assert torch.allclose(recon, G_expected, atol=1e-4), (
                f"tau={tau}, client={i}: FedSAA(lambda=1) did not reduce to FedAvg-on-merged-deltas"
            )

    print("PASS: lambda=1 (any tau) reproduces FedAvg-on-merged-deltas for every client.")


def test_lambda0_is_pure_self_attention_and_differs_across_clients():
    """Sanity check the other end of the blend: lambda=0 uses only the attended aggregate Ubar_i,
    which generally differs per client (unlike lambda=1's shared G)."""
    n_clients, r, out_ch, in_ch, k = 4, 4, 8, 6, 3
    states = _random_adapter_state(n_clients, r, out_ch, in_ch, k, seed=1)

    new_states, _ = fedsaa_aggregate(states, tau=0.1, lam=0.0, svd_rank=r)
    reconstructions = []
    for i in range(n_clients):
        A_new = new_states[i]["layer0.lora_A.default.weight"].reshape(r, -1)
        B_new = new_states[i]["layer0.lora_B.default.weight"].reshape(out_ch, r)
        reconstructions.append(B_new @ A_new)

    # not all clients should be identical when lambda=0 with a sharp-ish tau
    all_same = all(torch.allclose(reconstructions[0], r_i, atol=1e-4) for r_i in reconstructions[1:])
    assert not all_same, "lambda=0 unexpectedly produced identical adapters for every client"

    print("PASS: lambda=0 produces personalized (non-identical) per-client adapters.")


if __name__ == "__main__":
    test_lambda1_matches_fedavg_on_merged_deltas()
    test_lambda0_is_pure_self_attention_and_differs_across_clients()
    print("All FedSAA tests passed.")
