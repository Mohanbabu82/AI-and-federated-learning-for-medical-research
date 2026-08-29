"""One function per figure (F1-F7). Each reads directly from results/<run>/metrics.csv (or the
similarity/CKA .npz/.npy dumps) and returns a matplotlib Figure -- generate_figures.py handles
saving to paper/figures/.
"""

from __future__ import annotations

import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import METHOD_STYLE, OKABE_ITO


def _read_csv_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def plot_convergence(method_csv_paths: dict[str, str]) -> plt.Figure:
    """F1: accuracy vs round, one line per method."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, csv_path in method_csv_paths.items():
        rows = _read_csv_rows(csv_path)
        rounds = [int(r["round"]) for r in rows]
        acc = [float(r["global_acc"]) for r in rows]
        style = METHOD_STYLE[method]
        ax.plot(rounds, acc, **style)
    ax.set_xlabel("Round")
    ax.set_ylabel("Global test accuracy")
    ax.set_title("F1: Convergence")
    ax.set_xticks(sorted({int(r["round"]) for p in method_csv_paths.values() for r in _read_csv_rows(p)}))
    ax.legend()
    return fig


def plot_accuracy_vs_comm(method_csv_paths: dict[str, str]) -> plt.Figure:
    """F2: accuracy vs cumulative communication (MB)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, csv_path in method_csv_paths.items():
        rows = _read_csv_rows(csv_path)
        mb = [float(r["cumulative_MB"]) for r in rows]
        acc = [float(r["global_acc"]) for r in rows]
        style = METHOD_STYLE[method]
        if all(m == 0 for m in mb):
            # centralized/local_only never communicate -- show as a horizontal reference line
            ax.axhline(acc[-1], color=style["color"], linestyle=style["linestyle"],
                       label=f"{style['label']} (no communication)")
        else:
            ax.plot(mb, acc, **style)
    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel("Global test accuracy")
    ax.set_title("F2: Accuracy vs. communication cost")
    ax.legend()
    return fig


def plot_per_specialty_bars(method_csv_paths: dict[str, str], specialties: list[str]) -> plt.Figure:
    """F3: grouped bar chart, one group per specialty, one bar per method (final round)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = list(method_csv_paths.keys())
    n_methods = len(methods)
    n_specialties = len(specialties)
    x = np.arange(n_specialties)
    width = 0.8 / n_methods

    for i, method in enumerate(methods):
        rows = _read_csv_rows(method_csv_paths[method])
        final_row = rows[-1]
        per_client = [float(a) for a in final_row["per_client_acc"].split(";")]
        style = METHOD_STYLE[method]
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, per_client, width=width, color=style["color"], label=style["label"])

    ax.set_xticks(x)
    ax.set_xticklabels(specialties, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy")
    ax.set_title("F3: Per-specialty accuracy (final round)")
    ax.legend()
    return fig


def plot_similarity_heatmap(npz_path: str, layer_name: str | None = None) -> plt.Figure:
    """F4: cosine similarity matrix S heatmap for one LoRA layer (mean over layers if none given)."""
    data = np.load(npz_path)
    if layer_name is not None and layer_name in data:
        S = data[layer_name]
        title_suffix = layer_name.split(".")[-1]
    else:
        S = np.mean([data[k] for k in data.files], axis=0)
        title_suffix = f"mean over {len(data.files)} layers"

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(S, cmap="viridis", vmin=-1, vmax=1)
    n = S.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{i}" for i in range(n)])
    ax.set_yticklabels([f"C{i}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center",
                     color="white" if S[i, j] < 0.6 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    ax.set_title(f"F4: FedSAA client similarity S\n({title_suffix})")
    return fig


def plot_lambda_ablation(ablation_csv: str) -> plt.Figure:
    """F5: accuracy vs lambda (holding tau/rank/client_fraction at their most-sampled values)."""
    rows = _read_csv_rows(ablation_csv)
    # pick the (tau, rank, client_fraction) combo with the most lambda points sampled
    from collections import Counter
    combo_counts = Counter((r["tau"], r["rank"], r["client_fraction"]) for r in rows)
    tau, rank, frac = combo_counts.most_common(1)[0][0]
    subset = [r for r in rows if (r["tau"], r["rank"], r["client_fraction"]) == (tau, rank, frac)]
    subset.sort(key=lambda r: float(r["lambda"]))

    lambdas = [float(r["lambda"]) for r in subset]
    acc = [float(r["global_acc_mean"]) for r in subset]
    std = [float(r["global_acc_std"]) for r in subset]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(lambdas, acc, yerr=std, color=OKABE_ITO["bluish_green"], marker="o",
                capsize=4, linewidth=2)
    ax.set_xlabel(r"$\lambda$  (0 = pure attention, 1 = FedAvg-on-merged-deltas)")
    ax.set_ylabel("Global test accuracy")
    ax.set_title(f"F5: FedSAA lambda ablation\n(tau={tau}, rank={rank}, C={frac})")
    return fig


def plot_rank_ablation(ablation_csv: str) -> plt.Figure:
    """F6: accuracy vs LoRA rank r (holding lambda/tau/client_fraction at their most-sampled
    values), with adapter payload size on a secondary axis."""
    rows = _read_csv_rows(ablation_csv)
    from collections import Counter
    combo_counts = Counter((r["lambda"], r["tau"], r["client_fraction"]) for r in rows)
    lam, tau, frac = combo_counts.most_common(1)[0][0]
    subset = [r for r in rows if (r["lambda"], r["tau"], r["client_fraction"]) == (lam, tau, frac)]
    subset.sort(key=lambda r: int(r["rank"]))

    ranks = [int(r["rank"]) for r in subset]
    acc = [float(r["global_acc_mean"]) for r in subset]
    std = [float(r["global_acc_std"]) for r in subset]
    mb = [float(r["cumulative_MB_mean"]) for r in subset]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(ranks, acc, yerr=std, color=OKABE_ITO["blue"], marker="D", capsize=4, linewidth=2,
                label="accuracy")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Global test accuracy")
    ax.set_xticks(ranks)

    ax2 = ax.twinx()
    ax2.plot(ranks, mb, color=OKABE_ITO["vermillion"], marker="^", linestyle="--",
              label="cumulative comm. (MB)")
    ax2.set_ylabel("Cumulative communication (MB)")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax.set_title(f"F6: FedSAA rank ablation\n(lambda={lam}, tau={tau}, C={frac})")
    return fig


def plot_cka_collapse(cka_dirs: dict[str, str]) -> plt.Figure:
    """F7: mean pairwise client-CKA vs round, comparing methods (e.g. FedSAA vs FedAvg) --
    a representation-collapse indicator (higher = clients' features more homogenized)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for method, cka_dir in cka_dirs.items():
        files = sorted(glob.glob(os.path.join(cka_dir, "round_*.npy")))
        if not files:
            continue
        rounds, mean_cka = [], []
        for fp in files:
            round_idx = int(os.path.basename(fp).replace("round_", "").replace(".npy", ""))
            mat = np.load(fp)
            n = mat.shape[0]
            off_diag = mat[~np.eye(n, dtype=bool)]
            rounds.append(round_idx)
            mean_cka.append(float(np.nanmean(off_diag)))
        style = METHOD_STYLE[method]
        ax.plot(rounds, mean_cka, **style)

    ax.set_xlabel("Round")
    ax.set_ylabel("Mean pairwise client CKA")
    ax.set_title("F7: Representation collapse (FedSAA vs FedAvg)")
    ax.set_ylim(0, 1)
    ax.legend()
    return fig
