#!/usr/bin/env python
"""Completeness check for Integrated Gradients with the step count used in the
experiments (20) versus Captum's default (50).

IG's attributions should sum to f(x) - f(baseline) for the explained logit
(the completeness axiom). Captum returns the gap as the "convergence delta". We
report it relative to |f(x) - f(baseline)|, on the first --n-images of the same
seeded image selection run_experiment.py uses, for all 12 checkpoints.

Run from src/: python ig_convergence.py --n-images 100
"""
import argparse
import json
import os

import numpy as np
import torch
from captum.attr import IntegratedGradients

from saco.data import get_dataset
from saco.models import load_model

COMBOS = [(a, d) for a in ["vit_b16", "vit_l16", "deit_b16"]
          for d in ["imagenette", "imagenet", "cifar10", "cifar100"]]

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="../data")
p.add_argument("--n-images", type=int, default=100)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--steps", type=int, nargs="+", default=[20, 50])
p.add_argument("--internal-batch-size", type=int, default=None,
               help="Process IG steps in chunks of this size (same result, less GPU memory)")
p.add_argument("--combos", nargs="+", default=None, help="e.g. vit_l16_imagenet (default: all 12)")
p.add_argument("--output", default="../analysis/ig_convergence.json")
args = p.parse_args()
STEPS = args.steps
if args.combos:
    COMBOS = [c for c in COMBOS if f"{c[0]}_{c[1]}" in args.combos]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

out = {}
for arch, dataset in COMBOS:
    bundle = load_model(arch, dataset, None, device)
    ds = get_dataset(dataset, args.data_root, bundle.transform, split="val")
    # same selection as run_experiment.py (seed 42, 1000 images, sorted)
    rng = np.random.default_rng(args.seed)
    indices = sorted(rng.choice(len(ds), size=min(1000, len(ds)), replace=False).tolist())[:args.n_images]

    model = bundle.logits_module()
    ig = IntegratedGradients(model)
    rel = {s: [] for s in STEPS}
    for idx in indices:
        x = ds[idx][0].unsqueeze(0).to(device)
        baseline = torch.zeros_like(x)
        with torch.no_grad():
            logits = model(x)
            target = int(logits.argmax(1).item())
            gap = float(logits[0, target] - model(baseline)[0, target])
        for s in STEPS:
            _, delta = ig.attribute(x, baselines=baseline, target=target, n_steps=s,
                                    internal_batch_size=args.internal_batch_size,
                                    return_convergence_delta=True)
            rel[s].append(abs(float(delta.item())) / max(abs(gap), 1e-8))

    entry = {}
    for s in STEPS:
        r = np.array(rel[s])
        entry[f"steps_{s}"] = {"median_rel_error": float(np.median(r)), "mean_rel_error": float(np.mean(r)),
                               "p90_rel_error": float(np.percentile(r, 90)),
                               "frac_below_5pct": float(np.mean(r < 0.05))}
    out[f"{arch}_{dataset}"] = entry
    print(f"{arch}/{dataset}: " + " | ".join(
        f"{s} steps: median {entry[f'steps_{s}']['median_rel_error']:.3f}, p90 {entry[f'steps_{s}']['p90_rel_error']:.3f}"
        for s in STEPS), flush=True)
    del bundle, model, ig
    if device.type == "cuda":
        torch.cuda.empty_cache()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {args.output}")
