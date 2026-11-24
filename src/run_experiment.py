#!/usr/bin/env python
"""Runs the SaCo perturbation-sensitivity experiment for one (arch, dataset) pair.

Ported from saco_deit_analysis.ipynb, generalized to run any of {vit_b16, vit_l16,
deit_b16} against any of {imagenette, imagenet, cifar10, cifar100}. This is the one
script the SLURM job array fans out over -- see jobs/run_experiment.job.

Example:
    python run_experiment.py --arch deit_b16 --dataset imagenette \\
        --data-root ./data --output-dir ./results --num-images 200
"""
import argparse
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from saco.models import load_model, ARCHES
from saco.data import get_dataset
from saco.explanations import METHODS, ig_explanation
from saco.perturbations import build_strategies
from saco.metric import analyse_image
from saco.checkpoint import load_checkpoint, save_checkpoint

warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=ARCHES)
    p.add_argument("--dataset", required=True,
                    choices=["imagenette", "imagenet", "cifar10", "cifar100"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--checkpoint", default=None,
                    help="For imagenet/imagenette: a local torchvision/timm "
                         "state-dict (.pt) path, e.g. a perturbation-robustness "
                         "fine-tune, loaded on top of the standard pretrained "
                         "weights. For cifar10/cifar100: either an HF Hub repo id "
                         "to override the CIFAR checkpoint registry, or a local "
                         "state-dict (.pt) path (e.g. a perturbation-robustness "
                         "fine-tune), loaded on top of the registry checkpoint.")
    p.add_argument("--split", default="val")
    p.add_argument("--output-dir", default="./results")
    p.add_argument("--num-images", type=int, default=1000)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ig-steps", type=int, default=20)
    p.add_argument("--ig-internal-batch-size", type=int, default=None,
                    help="Process IG steps in chunks of this size: same result, less GPU memory")
    p.add_argument("--methods", nargs="+", default=None,
                    help="Only run these explanation methods (default: all). Used to "
                         "re-run a single method, e.g. IG at more integration steps, on "
                         "the same seeded images without recomputing the others.")
    p.add_argument("--blur-sigma", type=float, default=10)
    p.add_argument("--noise-std", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=1,
                    help="Write the checkpoint to disk every N images")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    bundle = load_model(args.arch, args.dataset, args.checkpoint, device)
    dataset = get_dataset(args.dataset, args.data_root, bundle.transform, split=args.split)
    print(f"Model: {args.arch} | Dataset: {args.dataset} ({len(dataset)} images)")

    rng = np.random.default_rng(args.seed)
    n = min(args.num_images, len(dataset))
    indices = sorted(rng.choice(len(dataset), size=n, replace=False).tolist())

    strategies = build_strategies(blur_sigma=args.blur_sigma, noise_std=args.noise_std)
    methods = dict(METHODS)
    methods["IG"] = partial(ig_explanation, n_steps=args.ig_steps,
                            internal_batch_size=args.ig_internal_batch_size)
    if args.methods:
        unknown = set(args.methods) - set(methods)
        if unknown:
            raise SystemExit(f"unknown method(s) {sorted(unknown)}; choose from {sorted(methods)}")
        methods = {m: fn for m, fn in methods.items() if m in args.methods}

    out_path = Path(args.output_dir) / f"{args.arch}_{args.dataset}" / "results.json"
    results = load_checkpoint(out_path)
    todo = [i for i in indices if str(i) not in results]
    print(f"{len(results)} done, {len(todo)} remaining -> {out_path}")

    classes = getattr(dataset, "classes", None)

    for count, img_idx in enumerate(tqdm(todo, desc=f"{args.arch}/{args.dataset}"), start=1):
        key = str(img_idx)
        x, y = dataset[img_idx]
        results[key] = {"true_label": classes[y] if classes else int(y)}

        for method_name, method_fn in methods.items():
            try:
                r = analyse_image(bundle, method_fn, x, strategies, K=args.k)
                results[key][method_name] = r
            except Exception as e:
                print(f"  Error - img {img_idx}, {method_name}: {e}")
                results[key][method_name] = {"scores": {s: None for s in strategies}}

        if count % args.save_every == 0:
            save_checkpoint(results, out_path)

    save_checkpoint(results, out_path)
    print(f"Done. {len(results)} images in {out_path}")


if __name__ == "__main__":
    main()
