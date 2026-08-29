"""Booktabs LaTeX + CSV table generation from results/summary/ and individual run dirs."""

from __future__ import annotations

import csv
import os

import numpy as np


def _latex_escape(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def fmt_mean_std(mean: float, std: float, decimals: int = 3, latex: bool = False) -> str:
    if np.isnan(mean):
        return "--"
    sep = r"$\pm$" if latex else "+/-"
    return f"{mean:.{decimals}f}{sep}{std:.{decimals}f}"


def write_csv(rows: list[dict], fieldnames: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_booktabs(headers: list[str], rows: list[list[str]], caption: str, label: str,
                    path: str, col_align: str | None = None) -> str:
    """Writes a booktabs-style LaTeX table (\\usepackage{booktabs} required) and returns the
    LaTeX source string as well.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    align = col_align or ("l" + "c" * (len(headers) - 1))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(_latex_escape(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    latex = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    return latex


def bold_best(values: list[float], higher_is_better: bool, decimals: int = 3,
              stds: list[float] | None = None) -> list[str]:
    """Formats a column of numbers, bolding (LaTeX \\textbf{}) the best entry. NaNs never win."""
    finite = [v for v in values if not np.isnan(v)]
    if not finite:
        best = None
    else:
        best = max(finite) if higher_is_better else min(finite)

    out = []
    for i, v in enumerate(values):
        text = fmt_mean_std(v, stds[i], decimals, latex=True) if stds is not None else (
            "--" if np.isnan(v) else f"{v:.{decimals}f}")
        if best is not None and not np.isnan(v) and v == best:
            text = rf"\textbf{{{text}}}"
        out.append(text)
    return out
