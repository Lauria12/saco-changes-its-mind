#!/usr/bin/env python
"""Is DeiT-B/16's register-token collapse driven by the weights or by the input?

debug_rollout_base_deit.py showed the un-fine-tuned base checkpoint already has
the patch-195 sink on (upsampled) CIFAR images, while the timm ImageNet DeiT-B/16
showed only 18% on Imagenette (Sec 6.1). This crosses three DeiT-B/16 weight sets
with two input distributions, each model using its own preprocessing:

  models: timm ImageNet DeiT-B/16 (used for Imagenette/ImageNet runs),
          HF base facebook/deit-base-patch16-224, HF CIFAR-10 fine-tune
  inputs: Imagenette val, CIFAR-10 test

Run from src/: python debug_rollout_deit_inputs.py --n-images 500
"""
import argparse
import json
import os
from collections import Counter

import torch

from saco.data import get_dataset
from saco.models import _load_hf_cifar, load_model

BASE = "facebook/deit-base-patch16-224"
REGISTER_TOKEN = 195

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="../data")
p.add_argument("--n-images", type=int, default=500)
p.add_argument("--output", default="../analysis/deit_inputs_mode_frequency.json")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

MODELS = {
    "timm_imagenet": lambda: load_model("deit_b16", "imagenette", None, device),
    "hf_base": lambda: _load_hf_cifar("deit_b16", "cifar10", BASE, device),
    "hf_cifar10_finetuned": lambda: load_model("deit_b16", "cifar10", None, device),
}


def rollout_top(bundle, ds, n):
    tops = []
    for i in range(n):
        x = ds[i][0].unsqueeze(0).to(device)
        attns = bundle.get_attentions(x, list(range(bundle.num_layers)))  # CPU tensors
        N = attns[0].shape[-1]
        result = torch.eye(N)
        for a in attns:
            a_res = (a[0].mean(0) + torch.eye(N)) / 2.0
            result = torch.matmul(a_res / a_res.sum(dim=-1, keepdim=True), result)
        tops.append(result[0, 1:].argmax().item())
    c = Counter(tops)
    tok, cnt = c.most_common(1)[0]
    return {"n": n, "mode_token": tok, "mode_freq": cnt / n, "unique_tokens": len(c),
            "register_freq": c[REGISTER_TOKEN] / n, "top5": c.most_common(5)}


out = {}
for name, make in MODELS.items():
    bundle = make()
    for dataset in ["imagenette", "cifar10"]:
        ds = get_dataset(dataset, args.data_root, bundle.transform, split="val")
        s = rollout_top(bundle, ds, args.n_images)
        out[f"{name}/{dataset}"] = s
        print(f"{name:22} on {dataset:10}: Rollout mode={s['mode_token']:>3} freq={s['mode_freq']:.3f} "
              f"unique={s['unique_tokens']:>3}/{s['n']} tok{REGISTER_TOKEN}={s['register_freq']:.3f} "
              f"top5={s['top5']}", flush=True)
    del bundle
    if device.type == "cuda":
        torch.cuda.empty_cache()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {args.output}")
