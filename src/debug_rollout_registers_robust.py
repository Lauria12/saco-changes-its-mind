#!/usr/bin/env python
"""Sink-token diagnostic for the register-token F-Fidelity extension
(submit_registers_pilot.sh): does robustifying deit_b16 against blur/noise/shuffle
on CIFAR change Rollout's collapse onto a content-independent "register" token?

Same measurement as debug_rollout_deit_cifar2.py (mode frequency of the top
token in RawAtt's last-layer CLS row and in Rollout's final CLS row), run on the
baseline CIFAR checkpoints and on each robustified checkpoint. Reports it over the
first 60 val images (directly comparable to the paper's mode-frequency table) and over
a larger N, plus how often the baseline's dominant Rollout token (195) is the top
token specifically.

If the register account is right, the robustified checkpoints should keep a high
Rollout mode frequency even where their SaCo margins improved.

Run from src/: python debug_rollout_registers_robust.py --n-images 500
"""
import argparse
import json
import os
from collections import Counter

import torch

from saco.models import load_model
from saco.data import get_dataset

REGISTER_TOKEN = 195  # baseline Rollout's dominant token (Sec 6.1)
N_COMPARABLE = 60     # Sec 6.1's original sample size

p = argparse.ArgumentParser()
p.add_argument("--data-root", default="../data")
p.add_argument("--ckpt-dir", default="../checkpoints/robust_registers")
p.add_argument("--n-images", type=int, default=500)
p.add_argument("--output", default="../analysis/robust_registers/sink_token_diagnostic.json")
args = p.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def summarize(tokens):
    c = Counter(tokens)
    mode_tok, mode_n = c.most_common(1)[0]
    return {
        "n": len(tokens),
        "mode_token": mode_tok,
        "mode_freq": mode_n / len(tokens),
        "unique_tokens": len(c),
        "register_freq": c[REGISTER_TOKEN] / len(tokens),
        "top5": c.most_common(5),
    }


out = {}
for dataset in ["cifar10", "cifar100"]:
    conditions = [("baseline", None)] + [
        (f"robust_{s}", os.path.join(args.ckpt_dir, f"deit_b16_{dataset}_{s}_robust.pt"))
        for s in ["blur", "noise", "shuffle"]
    ]
    for cond, checkpoint in conditions:
        if checkpoint is not None and not os.path.isfile(checkpoint):
            raise FileNotFoundError(checkpoint)
        print(f"\n{'='*70}\ndeit_b16 / {dataset} / {cond}  ({checkpoint or 'registry default'})\n{'='*70}")
        bundle = load_model("deit_b16", dataset, checkpoint, device)
        ds = get_dataset(dataset, args.data_root, bundle.transform, split="val")

        rawatt_top, rollout_top = [], []
        for img_i in range(args.n_images):
            x, _ = ds[img_i]
            x = x.unsqueeze(0).to(device)

            # get_attentions() returns CPU tensors (detach().cpu())
            all_attns = bundle.get_attentions(x, list(range(bundle.num_layers)))
            rawatt_top.append(all_attns[-1][0].mean(0)[0, 1:].argmax().item())

            N = all_attns[0].shape[-1]
            result = torch.eye(N)
            for attn in all_attns:
                attn_res = (attn[0].mean(0) + torch.eye(N)) / 2.0
                attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
                result = torch.matmul(attn_res, result)
            rollout_top.append(result[0, 1:].argmax().item())

        entry = {
            "checkpoint": checkpoint,
            f"rawatt_first{N_COMPARABLE}": summarize(rawatt_top[:N_COMPARABLE]),
            f"rollout_first{N_COMPARABLE}": summarize(rollout_top[:N_COMPARABLE]),
            "rawatt_all": summarize(rawatt_top),
            "rollout_all": summarize(rollout_top),
        }
        out[f"{dataset}/{cond}"] = entry
        for k in [f"rawatt_first{N_COMPARABLE}", f"rollout_first{N_COMPARABLE}", "rawatt_all", "rollout_all"]:
            s = entry[k]
            print(f"  {k:>16}: mode={s['mode_token']:>3} freq={s['mode_freq']:.3f} "
                  f"unique={s['unique_tokens']:>3}/{s['n']} tok{REGISTER_TOKEN}={s['register_freq']:.3f} "
                  f"top5={s['top5']}")

        del bundle
        if device.type == "cuda":
            torch.cuda.empty_cache()

os.makedirs(os.path.dirname(args.output), exist_ok=True)
with open(args.output, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {args.output}\nDone.")
