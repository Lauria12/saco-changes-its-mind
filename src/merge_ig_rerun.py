#!/usr/bin/env python
"""Swaps re-run Integrated Gradients results (more integration steps, written by
jobs/submit_ig_rerun.sh to a separate results tree) into the main per-image
results, and updates only the IG entries of analysis/<arch>_<dataset>_summary.json.

Every other method, Random Attribution included, keeps its stored per-image
scores, so IG's paired margins are still taken against Random on exactly the same
images. As a guard, the script re-derives every non-IG p-value from the merged
data and stops if any differs from the stored summary. The original results.json
files are never modified; merged copies go to --merged-dir.

Run from src/: python merge_ig_rerun.py --rerun-dir ../results_ig50 --ig-steps 50
then: multiple_comparisons.py.
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from analyze_results import METHODS, STRATEGIES, aggregate, bootstrap_ci, hodges_lehmann

p = argparse.ArgumentParser()
p.add_argument("--results-dir", default="../results")
p.add_argument("--rerun-dir", required=True)
p.add_argument("--merged-dir", default="../results_merged_ig")
p.add_argument("--analysis-dir", default="../analysis")
p.add_argument("--ig-steps", type=int, required=True)
p.add_argument("--n-bootstrap", type=int, default=5000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
np.random.seed(args.seed)  # bootstrap_ci draws from np.random

for summary_path in sorted(glob.glob(os.path.join(args.analysis_dir, "*_summary.json"))):
    combo = os.path.basename(summary_path)[:-len("_summary.json")]
    orig_path = os.path.join(args.results_dir, combo, "results.json")
    rerun_path = os.path.join(args.rerun_dir, combo, "results.json")
    if not (os.path.isfile(orig_path) and os.path.isfile(rerun_path)):
        print(f"skip {combo}: missing {orig_path if not os.path.isfile(orig_path) else rerun_path}")
        continue
    orig, rerun = json.load(open(orig_path)), json.load(open(rerun_path))

    # Only images with every method enter the statistics (aggregate()'s complete-case
    # filter); the rest are carry-overs from earlier, smaller runs and are left as-is.
    analysed = [k for k, v in orig.items() if all(m in v for m in METHODS)]
    missing = [k for k in analysed if k not in rerun]
    if missing:
        raise SystemExit(f"{combo}: {len(missing)} analysed images have no IG re-run (e.g. {missing[:3]})")
    for k in analysed:
        new_ig = rerun[k].get("IG")
        if new_ig is None or any(v is None for v in new_ig.get("scores", {}).values()):
            raise SystemExit(f"{combo}: image {k} has no complete IG re-run result")
        if new_ig["target_class"] != orig[k]["IG"]["target_class"]:
            raise SystemExit(f"{combo}: image {k} predicted class differs between runs")
        orig[k]["IG"] = new_ig

    os.makedirs(os.path.join(args.merged_dir, combo), exist_ok=True)
    with open(os.path.join(args.merged_dir, combo, "results.json"), "w") as f:
        json.dump(orig, f)

    summary = json.load(open(summary_path))
    agg, n_images = aggregate(orig)
    for method in METHODS:
        if method == "Random":
            continue
        for strat in STRATEGIES:
            m_scores = np.array(agg[method][strat])
            r_scores = np.array(agg["Random"][strat][:len(m_scores)])
            margins = m_scores - r_scores
            _, p_val = wilcoxon(margins)
            if method != "IG":
                if not np.isclose(p_val, summary[method][strat]["p_value"], rtol=1e-9, atol=0):
                    raise SystemExit(f"{combo} {method} {strat}: non-IG p-value changed after merge")
                continue
            old = summary["IG"][strat]
            ci_lo, ci_hi = bootstrap_ci(margins.tolist(), args.n_bootstrap)
            hl, hl_lo, hl_hi = hodges_lehmann(margins)
            summary["IG"][strat] = {
                "raw_mean": float(np.mean(m_scores)), "margin_mean": float(np.mean(margins)),
                "ci": [ci_lo, ci_hi], "p_value": float(p_val), "inversion": bool(np.mean(margins) < 0),
                "hl": hl, "hl_ci": [hl_lo, hl_hi], "ig_steps": args.ig_steps,
                "ig20_margin_mean": old.get("ig20_margin_mean", old["margin_mean"]),
                "ig20_p_value": old.get("ig20_p_value", old["p_value"]),
            }
            print(f"  {combo:22} IG {strat:8} margin {old['margin_mean']:+.4f} -> {np.mean(margins):+.4f}  "
                  f"p {old['p_value']:.2g} -> {p_val:.2g}")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{combo}: n={n_images}, IG entries updated ({args.ig_steps} steps)")
