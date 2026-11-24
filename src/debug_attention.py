#!/usr/bin/env python
"""Diagnostic: visualize raw last-layer attention for vit_l16 vs vit_b16 on a
few Imagenette images, to sanity-check why vit_l16's RawAtt SaCo score came
back as exactly 0.0000 across every perturbation strategy in the main run.

Reuses the same load_model()/get_attentions() code path run_experiment.py uses,
so this checks the actual pipeline rather than a reimplementation.

Example:
    python debug_attention.py --data-root ../data --output-dir ../analysis/attn_debug
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from saco.models import load_model
from saco.data import get_dataset


def cls_to_patch_attention(bundle, x, layer_idx):
    attns = bundle.get_attentions(x, layer_indices=[layer_idx])
    attn = attns[0]  # [B, heads, N, N]
    attn = attn.mean(dim=1)  # average over heads -> [B, N, N]
    cls_row = attn[0, 0, 1:]  # CLS attending to patches, drop CLS->CLS entry
    return cls_row.numpy()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="../data")
    p.add_argument("--n-images", type=int, default=4)
    p.add_argument("--output-dir", default="../analysis/attn_debug")
    args = p.parse_args()

    device = torch.device("cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for arch in ["vit_b16", "vit_l16"]:
        print(f"=== {arch} ===")
        bundle = load_model(arch, "imagenette", checkpoint=None, device=device)
        dataset = get_dataset("imagenette", args.data_root, bundle.transform, split="val")

        last_layer = bundle.num_layers - 1
        sweep_layers = sorted(set([0, bundle.num_layers // 2, last_layer]))
        print(f"  num_layers={bundle.num_layers}, layer sweep={sweep_layers}")

        images = [dataset[i][0] for i in range(args.n_images)]

        for layer_idx in sweep_layers:
            print(f"  -- layer {layer_idx} --")
            for i, img in enumerate(images):
                x = img.unsqueeze(0).to(device)
                cls_row = cls_to_patch_attention(bundle, x, layer_idx)
                print(f"    image {i}: mean={cls_row.mean():.6f} std={cls_row.std():.6f} "
                      f"min={cls_row.min():.6f} max={cls_row.max():.6f} sum={cls_row.sum():.6f}")

        fig, axes = plt.subplots(2, args.n_images, figsize=(4 * args.n_images, 8))
        for i, img in enumerate(images):
            x = img.unsqueeze(0).to(device)
            cls_row = cls_to_patch_attention(bundle, x, last_layer)

            grid_size = int(round(np.sqrt(len(cls_row))))
            heatmap = cls_row[: grid_size * grid_size].reshape(grid_size, grid_size)

            img_np = img.permute(1, 2, 0).numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

            axes[0, i].imshow(img_np)
            axes[0, i].set_title(f"image {i}")
            axes[0, i].axis("off")

            axes[1, i].imshow(img_np)
            axes[1, i].imshow(heatmap, cmap="jet", alpha=0.5,
                               extent=(0, img_np.shape[1], img_np.shape[0], 0))
            axes[1, i].set_title(f"layer {last_layer}, std={cls_row.std():.5f}")
            axes[1, i].axis("off")

        plt.tight_layout()
        out_path = out_dir / f"{arch}_raw_attention_debug.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}\n")


if __name__ == "__main__":
    main()
