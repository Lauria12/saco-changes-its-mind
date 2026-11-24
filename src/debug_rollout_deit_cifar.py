#!/usr/bin/env python
"""Diagnostic: why does Rollout fail specifically on deit_b16 x {cifar10, cifar100}
(significantly below random on 4/5 strategies) while working fine on deit_b16 x
imagenette, and while RawAtt (last layer only) is actually FINE on deit_b16/cifar10?

Since RawAtt (last layer alone) works but Rollout (all-layer product) fails, the
problem must be introduced by early/middle layers that Rollout aggregates but
RawAtt ignores. Captures per-layer attention for all 12 layers across the 3
checkpoints and looks specifically for "attention sink" tokens: a token position
that receives disproportionate attention mass from most/all query rows in a given
layer (not just from CLS) -- a well-documented pathology in fine-tuned transformers
that would explain the final Rollout CLS-row column ending up dominated by a token
uncorrelated with the actually-salient image region.
"""
import sys
from pathlib import Path

import numpy as np
import torch

from saco.models import load_model
from saco.data import get_dataset

DATA_ROOT = "../data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

COMBOS = [
    ("imagenette", None, "WORKS (ImageNet-pretrained, timm)"),
    ("cifar10", None, "FAILS (fine-tuned in-house)"),
    ("cifar100", None, "FAILS (fine-tuned in-house)"),
]

N_IMAGES = 4


def analyze(bundle, x):
    """Returns per-layer stats: cls_row std/max/argmax, and column-sum
    (attention received from ALL queries, not just CLS) max/argmax per layer."""
    num_layers = bundle.num_layers
    attns = bundle.get_attentions(x, list(range(num_layers)))
    stats = []
    for i, attn in enumerate(attns):
        a = attn[0].mean(0)  # [N, N] heads-averaged
        cls_row = a[0, 1:]  # CLS attending to patches
        col_sum = a[:, 1:].sum(dim=0)  # total incoming attention per patch, from ALL queries (incl CLS)
        stats.append({
            "layer": i,
            "cls_row_std": cls_row.std().item(),
            "cls_row_max": cls_row.max().item(),
            "cls_row_argmax": cls_row.argmax().item(),
            "colsum_max": col_sum.max().item(),
            "colsum_argmax": col_sum.argmax().item(),
            "colsum_mean": col_sum.mean().item(),
            "colsum_max_over_mean": (col_sum.max() / (col_sum.mean() + 1e-8)).item(),
        })
    return stats


for dataset, checkpoint, label in COMBOS:
    print(f"\n{'='*70}\ndeit_b16 / {dataset} -- {label}\n{'='*70}")
    bundle = load_model("deit_b16", dataset, checkpoint, device)
    ds = get_dataset(dataset, DATA_ROOT, bundle.transform, split="val")

    for img_i in range(N_IMAGES):
        x, y = ds[img_i]
        x = x.unsqueeze(0).to(device)
        stats = analyze(bundle, x)
        print(f"\n  -- image {img_i} --")
        print(f"  {'layer':>5} {'cls_std':>9} {'cls_max':>9} {'cls_argmax':>11} "
              f"{'colsum_max':>11} {'colsum_argmax':>14} {'max/mean':>9}")
        for s in stats:
            print(f"  {s['layer']:>5} {s['cls_row_std']:>9.5f} {s['cls_row_max']:>9.5f} "
                  f"{s['cls_row_argmax']:>11} {s['colsum_max']:>11.4f} "
                  f"{s['colsum_argmax']:>14} {s['colsum_max_over_mean']:>9.2f}")

    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nDone.")
