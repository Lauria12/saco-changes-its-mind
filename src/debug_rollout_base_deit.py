#!/usr/bin/env python
"""Does CIFAR fine-tuning create DeiT-B/16's register token, or reveal one that the
base checkpoint already has?

Runs the Sec 6.1 mode-frequency test on the un-fine-tuned base checkpoint
(facebook/deit-base-patch16-224, the starting point of both in-house CIFAR
fine-tunes), on the SAME preprocessed CIFAR inputs as the fine-tuned checkpoints:
both get the fine-tuned checkpoint's own transform, so only the weights differ.
The fine-tuned checkpoints are re-run alongside as the reference.

Run from src/: python debug_rollout_base_deit.py --n-images 500
"""
import argparse
import json
import os
from collections import Counter

import torch

from saco.data import get_dataset
from saco.models import load_model

BASE = "facebook/deit-base-patch16-224"
REGISTER_TOKEN = 195
N_COMPARABLE = 60

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="../data")
p.add_argument("--n-images", type=int, default=500)
p.add_argument("--output", default="../analysis/base_deit_mode_frequency.json")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def top_tokens(bundle, ds, n):
    rawatt, rollout = [], []
    for i in range(n):
        x = ds[i][0].unsqueeze(0).to(device)
        attns = bundle.get_attentions(x, list(range(bundle.num_layers)))  # CPU tensors
        rawatt.append(attns[-1][0].mean(0)[0, 1:].argmax().item())
        N = attns[0].shape[-1]
        result = torch.eye(N)
        for a in attns:
            a_res = (a[0].mean(0) + torch.eye(N)) / 2.0
            result = torch.matmul(a_res / a_res.sum(dim=-1, keepdim=True), result)
        rollout.append(result[0, 1:].argmax().item())
    return rawatt, rollout


def summarize(tokens):
    c = Counter(tokens)
    tok, cnt = c.most_common(1)[0]
    return {"n": len(tokens), "mode_token": tok, "mode_freq": cnt / len(tokens),
            "unique_tokens": len(c), "register_freq": c[REGISTER_TOKEN] / len(tokens),
            "top5": c.most_common(5)}


out = {}
for dataset in ["cifar10", "cifar100"]:
    finetuned = load_model("deit_b16", dataset, None, device)
    ds = get_dataset(dataset, args.data_root, finetuned.transform, split="val")
    base = load_model("deit_b16", dataset, BASE, device)
    for name, bundle in [("finetuned", finetuned), ("base", base)]:
        rawatt, rollout = top_tokens(bundle, ds, args.n_images)
        entry = {f"rawatt_first{N_COMPARABLE}": summarize(rawatt[:N_COMPARABLE]),
                 f"rollout_first{N_COMPARABLE}": summarize(rollout[:N_COMPARABLE]),
                 "rawatt_all": summarize(rawatt), "rollout_all": summarize(rollout)}
        out[f"{dataset}/{name}"] = entry
        for k, s in entry.items():
            print(f"{dataset:9} {name:9} {k:>16}: mode={s['mode_token']:>3} freq={s['mode_freq']:.3f} "
                  f"unique={s['unique_tokens']:>3}/{s['n']} tok{REGISTER_TOKEN}={s['register_freq']:.3f} "
                  f"top5={s['top5']}", flush=True)
    del finetuned, base
    if device.type == "cuda":
        torch.cuda.empty_cache()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(out, f, indent=2)
print(f"Saved {args.output}")
