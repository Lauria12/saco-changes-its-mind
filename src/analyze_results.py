#!/usr/bin/env python
"""Aggregates one or more results.json files from run_experiment.py into the
summary table + plots from saco_deit_analysis.ipynb (bootstrap CI, Wilcoxon vs
Random, margin heatmap, rank-inversion scatter). Headless-safe (Agg backend, no
plt.show()) since this runs on a login node or a plotting-only job, not the GPU node
that produced the data.

Example:
    python analyze_results.py --results results/deit_b16_imagenette/results.json \\
        --output-dir ./analysis
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

METHODS = ["Random", "IG", "RawAtt", "Rollout", "TransMM"]
STRATEGIES = ["mean", "blur", "noise", "shuffle", "inpaint"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, nargs="+",
                    help="One or more results.json paths from run_experiment.py")
    p.add_argument("--label", default=None,
                    help="Name for this run in printed output (default: inferred from path)")
    p.add_argument("--output-dir", default="./analysis")
    p.add_argument("--n-bootstrap", type=int, default=5000)
    return p.parse_args()


def load_results(paths):
    merged = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        merged.update(data)
    return merged


def aggregate(results):
    # Only keep images that have every method present. Images carried over via
    # checkpointing from a run before a method existed (e.g. TransMM added later)
    # would otherwise have some methods missing -- silently dropping just those
    # methods for just those images breaks the positional pairing summarize() and
    # the Wilcoxon test rely on (index i of every method's list must be the same
    # image), since agg[method][strat] would end up different lengths/orders per
    # method instead of all being built from the same aligned image sequence.
    complete = {k: v for k, v in results.items() if all(m in v for m in METHODS)}
    dropped = len(results) - len(complete)
    if dropped:
        print(f"Note: dropped {dropped}/{len(results)} images missing one or more "
              f"methods (likely carried over via checkpointing from before a method "
              f"was added)")

    agg = {m: {s: [] for s in STRATEGIES} for m in METHODS}
    for img_data in complete.values():
        for method_name in METHODS:
            scores = img_data[method_name].get("scores", {})
            for strat in STRATEGIES:
                val = scores.get(strat)
                if val is not None:
                    agg[method_name][strat].append(val)
    return agg, len(complete)


def bootstrap_ci(scores, n_boot, ci=0.95):
    if len(scores) < 5:
        return float("nan"), float("nan")
    arr = np.array(scores)
    boot = [np.mean(np.random.choice(arr, len(arr), replace=True)) for _ in range(n_boot)]
    lo = (1 - ci) / 2 * 100
    return float(np.percentile(boot, lo)), float(np.percentile(boot, 100 - lo))


def hodges_lehmann(margins, ci=0.95):
    """Hodges-Lehmann location estimate and its distribution-free CI.

    This is the estimator that belongs to the Wilcoxon signed-rank test: the
    median of all Walsh averages (m_i + m_j) / 2, i <= j. Its CI is read off the
    sorted Walsh averages at the signed-rank critical value (normal approximation,
    which is accurate at n ~ 1000), so it excludes zero exactly when the
    two-sided Wilcoxon test rejects at 1 - ci, up to ties.
    """
    from scipy.stats import norm
    m = np.asarray(margins, dtype=float)
    n = len(m)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    i, j = np.triu_indices(n)
    walsh = (m[i] + m[j]) / 2.0
    M = len(walsh)
    z = norm.ppf(1 - (1 - ci) / 2)
    c = int(np.floor(n * (n + 1) / 4 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24)))
    lo_idx, hi_idx = max(c - 1, 0), min(M - c, M - 1)
    mid = (M - 1) // 2
    kth = sorted({lo_idx, hi_idx, mid, M // 2})
    part = np.partition(walsh, kth)
    estimate = float(part[mid]) if M % 2 else float((part[mid] + part[M // 2]) / 2)
    return estimate, float(part[lo_idx]), float(part[hi_idx])


def summarize(agg, n_images, n_bootstrap, label):
    random_scores = agg["Random"]
    summary = {}

    print(f"\n=== {label} ===")
    print(f"SaCo margins vs Random Attribution (n={n_images} images, 95% CI)\n")
    header = f"{'Method':<12} {'Strategy':<12} {'Raw SaCo':>10} {'Margin':>10}  {'95% CI':^22}  {'p-value':>9}  Note"
    print(header)
    print("-" * len(header))

    for method_name in METHODS:
        if method_name == "Random":
            continue
        summary[method_name] = {}
        for strat in STRATEGIES:
            m_scores = np.array(agg[method_name][strat])
            r_scores = np.array(random_scores[strat][:len(m_scores)])
            if len(m_scores) == 0 or len(m_scores) != len(r_scores):
                continue
            margins = m_scores - r_scores

            ci_lo, ci_hi = bootstrap_ci(margins.tolist(), n_bootstrap)
            raw_mean = float(np.mean(m_scores))
            margin_mean = float(np.mean(margins))

            p_val = float("nan")
            if len(margins) > 4:
                try:
                    _, p_val = wilcoxon(margins)
                except Exception:
                    pass

            inversion = margin_mean < 0
            ci_str = f"[{ci_lo:+.4f}, {ci_hi:+.4f}]"
            note = "<<< BELOW RANDOM" if inversion else ("  significant" if p_val < 0.05 else "")

            hl, hl_lo, hl_hi = hodges_lehmann(margins)
            summary[method_name][strat] = {
                "raw_mean": raw_mean, "margin_mean": margin_mean,
                "ci": [ci_lo, ci_hi], "p_value": p_val, "inversion": inversion,
                "hl": hl, "hl_ci": [hl_lo, hl_hi],
            }
            print(f"{method_name:<12} {strat:<12} {raw_mean:>+10.4f} {margin_mean:>+10.4f}  "
                  f"{ci_str:<22}  {p_val:>9.4f}  {note}")
        print()

    return summary


def plot_heatmap(summary, label, out_path):
    methods_list = [m for m in METHODS if m != "Random" and m in summary]
    margin_matrix = np.array([
        [summary[m].get(s, {}).get("margin_mean", np.nan) for s in STRATEGIES]
        for m in methods_list
    ])

    fig, ax = plt.subplots(figsize=(10, 4))
    vmax = max(0.1, np.nanmax(np.abs(margin_matrix)))
    im = ax.imshow(margin_matrix, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(STRATEGIES)))
    ax.set_xticklabels(STRATEGIES, rotation=25, ha="right", fontsize=11)
    ax.set_yticks(range(len(methods_list)))
    ax.set_yticklabels(methods_list, fontsize=11)

    for i, method_name in enumerate(methods_list):
        for j, strat in enumerate(STRATEGIES):
            val = margin_matrix[i, j]
            if np.isnan(val):
                continue
            inv = summary[method_name][strat]["inversion"]
            color = "white" if abs(val) > vmax * 0.5 else "black"
            mark = " !" if inv else ""
            ax.text(j, i, f"{val:+.3f}{mark}", ha="center", va="center",
                    fontsize=10, color=color, fontweight="bold" if inv else "normal")

    plt.colorbar(im, ax=ax, label="Mean margin vs Random Attribution")
    ax.set_title(f"{label} - SaCo margin over random baseline per method x strategy\n"
                 "Green = above random | Red = below random | ! = rank inversion", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label = args.label or Path(args.results[0]).parent.name
    results = load_results(args.results)
    agg, n_images = aggregate(results)

    summary = summarize(agg, n_images, args.n_bootstrap, label)
    plot_heatmap(summary, label, out_dir / f"{label}_margin_heatmap.png")

    with open(out_dir / f"{label}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
