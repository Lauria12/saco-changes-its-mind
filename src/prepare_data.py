#!/usr/bin/env python
"""Pre-fetches everything that needs internet access: datasets and model weights.

Run this on the `staging` partition (or the login node, if it's quick) before
submitting GPU jobs on gpu_a100/gpu_h100/gpu_mig -- those compute nodes are not
guaranteed to have internet access, so this must happen first, once, with results
cached under --data-root / the shared HF cache.

Example:
    python prepare_data.py --datasets imagenette cifar10 cifar100 \\
        --arches vit_b16 vit_l16 deit_b16 --data-root ./data
"""
import argparse
from pathlib import Path

import torch

from saco.models import load_model, ARCHES
from saco.data import get_dataset, stage_imagenet_val


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["imagenette"],
                    choices=["imagenette", "cifar10", "cifar100", "imagenet"],
                    help="imagenet stages the validation split only (~50k images), "
                         "read directly out of Snellius's shared Kaggle ImageNet "
                         "zip -- see saco.data.stage_imagenet_val")
    p.add_argument("--arches", nargs="+", default=list(ARCHES), choices=ARCHES)
    p.add_argument("--data-root", default="./data")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")  # staging nodes have no GPU -- this only downloads

    if "imagenet" in args.datasets:
        print("Staging ImageNet val (one-time, arch-independent)")
        stage_imagenet_val(Path(args.data_root))

    for arch in args.arches:
        for dataset in args.datasets:
            print(f"Fetching weights: {arch} / {dataset}")
            try:
                bundle = load_model(arch, dataset, checkpoint=None, device=device)
            except ValueError as e:
                print(f"  skipped: {e}")
                continue
            print(f"Fetching data: {dataset}")
            get_dataset(dataset, args.data_root, bundle.transform, split="val")

    print("Done.")


if __name__ == "__main__":
    main()
