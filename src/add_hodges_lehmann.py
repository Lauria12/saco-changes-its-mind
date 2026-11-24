#!/usr/bin/env python
"""Adds the Hodges-Lehmann estimate and CI ("hl", "hl_ci") to existing
analysis/<arch>_<dataset>_summary.json files, without recomputing anything else.

Re-running analyze_results.py would also redraw its (unseeded) bootstrap CIs and
shift every already-reported CI slightly, so this reuses its aggregation and
pairing logic and only fills in the new fields. It also re-runs the Wilcoxon test
on the same margins and refuses to write if any p-value differs from the stored
one, which guarantees the HL numbers come from exactly the same paired images.

Run from src/: python add_hodges_lehmann.py
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

from analyze_results import METHODS, STRATEGIES, aggregate, hodges_lehmann, load_results

p = argparse.ArgumentParser()
p.add_argument("--analysis-dir", default="../analysis")
p.add_argument("--results-dir", default="../results")
args = p.parse_args()

for summary_path in sorted(glob.glob(os.path.join(args.analysis_dir, "*_summary.json"))):
    combo = os.path.basename(summary_path)[:-len("_summary.json")]
    results_path = os.path.join(args.results_dir, combo, "results.json")
    if not os.path.isfile(results_path):
        print(f"skip {combo}: no {results_path}")
        continue
    summary = json.load(open(summary_path))
    agg, n_images = aggregate(load_results([results_path]))
    disagreements = 0
    for method in METHODS:
        if method == "Random":
            continue
        for strat in STRATEGIES:
            m_scores = np.array(agg[method][strat])
            r_scores = np.array(agg["Random"][strat][:len(m_scores)])
            margins = m_scores - r_scores
            entry = summary[method][strat]
            _, p_val = wilcoxon(margins)
            if not np.isclose(p_val, entry["p_value"], rtol=1e-9, atol=0):
                raise SystemExit(f"{combo} {method} {strat}: recomputed p={p_val} != stored {entry['p_value']}")
            hl, lo, hi = hodges_lehmann(margins)
            entry["hl"], entry["hl_ci"] = hl, [lo, hi]
            if (p_val < 0.05) != (lo > 0 or hi < 0):
                disagreements += 1
                print(f"  note: {combo} {method} {strat}: p={p_val:.4f}, HL CI=[{lo:+.4f}, {hi:+.4f}]")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"{combo}: n={n_images}, HL added, {disagreements} cells where p<0.05 and the HL CI disagree")
