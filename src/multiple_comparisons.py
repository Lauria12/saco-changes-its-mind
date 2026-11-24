#!/usr/bin/env python
"""Multiple-comparison correction across the full grid.

Each analysis/<arch>_<dataset>_summary.json holds one Wilcoxon p-value per
(method, strategy) cell. Correcting per combination would undercount the tests
actually run, so this pools all cells of the 12-combination grid (4 methods x
5 strategies x 12 combinations = 240 tests) and computes:

  - Benjamini-Hochberg adjusted p-values (q), the main significance rule;
  - Holm adjusted p-values, as a stricter family-wise robustness check.

Writes analysis/multiple_comparisons.json. With --update-md --results-md FILE it
also inserts a "q (BH)" column into the per-combination markdown tables of FILE.

Run from src/: python multiple_comparisons.py
"""
import argparse
import glob
import json
import os
import re

ARCH_NAMES = {"vit_b16": "ViT-B/16", "vit_l16": "ViT-L/16", "deit_b16": "DeiT-B/16"}
DATASET_NAMES = {"imagenette": "Imagenette", "imagenet": "ImageNet-1k",
                 "cifar10": "CIFAR-10", "cifar100": "CIFAR-100"}

p = argparse.ArgumentParser()
p.add_argument("--analysis-dir", default="../analysis")
p.add_argument("--results-md", default=None)
p.add_argument("--update-md", action="store_true")
args = p.parse_args()

cells = []
for path in sorted(glob.glob(os.path.join(args.analysis_dir, "*_summary.json"))):
    combo = os.path.basename(path)[:-len("_summary.json")]
    arch, dataset = next((a, combo[len(a) + 1:]) for a in ARCH_NAMES if combo.startswith(a + "_"))
    if dataset not in DATASET_NAMES:
        continue
    for method, by_strategy in json.load(open(path)).items():
        for strategy, v in by_strategy.items():
            cells.append(dict(arch=arch, dataset=dataset, method=method, strategy=strategy,
                              margin=v["margin_mean"], p=v["p_value"]))

n = len(cells)
order = sorted(range(n), key=lambda i: cells[i]["p"])

# Holm: step-down, adjusted p is the running max of (n - rank) * p
running = 0.0
for rank, i in enumerate(order):
    running = max(running, min(1.0, (n - rank) * cells[i]["p"]))
    cells[i]["p_holm"] = running

# Benjamini-Hochberg: step-up, adjusted p is the running min from the top of n / rank * p
running = 1.0
for rank in range(n - 1, -1, -1):
    i = order[rank]
    running = min(running, cells[i]["p"] * n / (rank + 1))
    cells[i]["q_bh"] = running

summary = {}
for method in sorted({c["method"] for c in cells}):
    mc = [c for c in cells if c["method"] == method]
    summary[method] = {
        rule: {"sig_faithful": sum(c[key] < 0.05 and c["margin"] > 0 for c in mc),
               "sig_below": sum(c[key] < 0.05 and c["margin"] < 0 for c in mc),
               "below_any": sum(c["margin"] < 0 for c in mc),
               "n_cells": len(mc)}
        for rule, key in [("uncorrected", "p"), ("bh", "q_bh"), ("holm", "p_holm")]
    }

out_path = os.path.join(args.analysis_dir, "multiple_comparisons.json")
with open(out_path, "w") as f:
    json.dump({"n_tests": n, "counts": summary, "cells": cells}, f, indent=2)
print(f"{n} tests -> {out_path}")
for method, s in summary.items():
    print(f"  {method:8} " + "  ".join(f"{r}: +{v['sig_faithful']}/-{v['sig_below']}" for r, v in s.items()))


def fmt_q(q):
    return "<0.0001" if q < 0.0001 else f"{q:.4f}"


if args.update_md:
    lookup = {(ARCH_NAMES[c["arch"]], DATASET_NAMES[c["dataset"]], c["method"], c["strategy"].capitalize()): c
              for c in cells}
    lines = open(args.results_md, encoding="utf-8").read().split("\n")
    combo, method, n_updated = None, None, 0
    for k, ln in enumerate(lines):
        m = re.match(r"^####\s+(.+?)\s+/\s+(.+?)\s+\(n=\d+\)", ln.strip())
        if m:
            combo, method = (m.group(1), m.group(2)), None
            continue
        if combo is None or not ln.startswith("|"):
            if ln.startswith("#"):
                combo = None
            continue
        parts = ln.split("|")  # ['', method, strategy, raw, margin, ci, p, (q,) flag, '']
        body = [c.strip() for c in parts[1:-1]]
        if body[0] == "Method":
            if "q (BH)" not in body:
                body.insert(6, "q (BH)")
            lines[k] = "| " + " | ".join(body) + " |"
            continue
        if set(body[0]) <= {"-", ":"} and body[0]:
            if len(body) == 7:
                body.insert(6, "---:")
            lines[k] = "|" + "|".join(body) + "|"
            continue
        if body[0]:
            method = body[0]
        cell = lookup[(combo[0], combo[1], method, body[1])]
        q = fmt_q(cell["q_bh"])
        if len(body) == 7:
            body.insert(6, q)
        else:
            body[6] = q
        lines[k] = "| " + " | ".join(body) + " |"
        n_updated += 1
    open(args.results_md, "w", encoding="utf-8").write("\n".join(lines))
    print(f"updated {n_updated} table rows in {args.results_md}")
