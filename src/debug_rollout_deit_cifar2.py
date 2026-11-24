#!/usr/bin/env python
"""Follow-up to debug_rollout_deit_cifar.py: is the last-layer attention sink token
MORE content-independent (i.e., the same token position dominates regardless of
image content) on the CIFAR fine-tuned deit_b16 checkpoints than on the
ImageNet-pretrained one? If fine-tuning induced/amplified a fixed "register" token,
we'd expect a much higher mode-frequency (same argmax token recurring across many
different images) for cifar10/cifar100 than for imagenette.

Checks both the LAST layer's CLS row (what RawAtt uses) and the final Rollout
CLS row (what Rollout uses, after the full 12-layer product) across N images.
"""
import sys
from collections import Counter

import torch

from saco.models import load_model
from saco.data import get_dataset
from saco.explanations import rollout_explanation

DATA_ROOT = "../data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

N_IMAGES = 60
COMBOS = [("imagenette", "ImageNet-pretrained (WORKS)"), ("cifar10", "fine-tuned (FAILS)"),
          ("cifar100", "fine-tuned (FAILS)")]


def top_k_mode(counter, k=5):
    return counter.most_common(k)


for dataset, label in COMBOS:
    print(f"\n{'='*70}\ndeit_b16 / {dataset} -- {label}\n{'='*70}")
    bundle = load_model("deit_b16", dataset, None, device)
    ds = get_dataset(dataset, DATA_ROOT, bundle.transform, split="val")

    last_layer_argmax = Counter()
    rollout_grid_argmax = Counter()

    for img_i in range(N_IMAGES):
        x, y = ds[img_i]
        x = x.unsqueeze(0).to(device)

        attns = bundle.get_attentions(x, [bundle.num_layers - 1])
        a = attns[0][0].mean(0)
        cls_row = a[0, 1:]
        last_layer_argmax[cls_row.argmax().item()] += 1

        # Also track the argmax of the pre-upsample rollout CLS row directly
        # (bypass _cls_to_patch_grid's upsampling to keep patch-index granularity)
        all_attns = bundle.get_attentions(x, list(range(bundle.num_layers)))
        N = all_attns[0].shape[-1]
        result = torch.eye(N)  # get_attentions() always returns CPU tensors (detach().cpu())
        for attn in all_attns:
            attn_mean = attn[0].mean(0)
            attn_res = (attn_mean + torch.eye(N)) / 2.0
            attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
            result = torch.matmul(attn_res, result)
        rollout_cls = result[0, 1:]
        rollout_grid_argmax[rollout_cls.argmax().item()] += 1

    n = N_IMAGES
    print(f"\n  Last-layer (RawAtt) CLS-row argmax distribution over {n} images:")
    top = top_k_mode(last_layer_argmax)
    print(f"    top tokens: {top}")
    print(f"    mode frequency (most common token / n): {top[0][1]/n:.3f}" if top else "    (empty)")
    print(f"    unique tokens used: {len(last_layer_argmax)} / {n} images")

    print(f"\n  Final Rollout CLS-row argmax distribution over {n} images:")
    top = top_k_mode(rollout_grid_argmax)
    print(f"    top tokens: {top}")
    print(f"    mode frequency (most common token / n): {top[0][1]/n:.3f}" if top else "    (empty)")
    print(f"    unique tokens used: {len(rollout_grid_argmax)} / {n} images")

    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

print("\nDone.")
