#!/usr/bin/env python
"""F-Fidelity-inspired "explanation-agnostic" robustness fine-tuning (arXiv:2410.02970).

Tests whether the perturbation-strategy-dependent instability found in Raw
Attention/Rollout's SaCo faithfulness verdict is (at least partly) a reaction to
the perturbation's out-of-distribution visual artifact, rather than a genuine
property of the explanation method. Fine-tunes a model to reproduce its OWN
clean-image top-1 prediction (a frozen pseudo-label, not ground truth -- avoids
any label-space remapping issues) on randomly-masked-and-perturbed versions of
training images, using ONE specific perturbation strategy at a time. If
re-evaluating SaCo against that same strategy on the robustified model removes or
shrinks the instability, that supports the OOD-artifact hypothesis; if it
persists, that rules it out.

The mask is a random contiguous rectangle, NOT saliency-guided (no explanation is
computed during fine-tuning) -- the goal is general robustness to the
perturbation's appearance, matching F-Fidelity's "random masking" framing.

Example:
    python finetune_perturbation_robust.py --arch vit_b16 --dataset imagenette \\
        --strategy blur --data-root ../data --output-dir ../checkpoints/robust
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from saco.models import load_model
from saco.data import get_dataset
from saco.perturbations import build_strategies


def random_mask(H, W, rng, min_frac=0.1, max_frac=0.4):
    frac = rng.uniform(min_frac, max_frac)
    area = frac * H * W
    aspect = rng.uniform(0.5, 2.0)
    h = int(round(np.sqrt(area * aspect)))
    w = int(round(np.sqrt(area / aspect)))
    h = min(max(h, 1), H)
    w = min(max(w, 1), W)
    top = int(rng.integers(0, H - h + 1))
    left = int(rng.integers(0, W - w + 1))
    mask = np.zeros((H, W), dtype=bool)
    mask[top:top + h, left:left + w] = True
    return mask


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--strategy", required=True,
                    choices=["mean", "blur", "noise", "shuffle", "inpaint"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./checkpoints/robust")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--n-train-images", type=int, default=2000)
    p.add_argument("--blur-sigma", type=float, default=10)
    p.add_argument("--noise-std", type=float, default=0.3)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    bundle = load_model(args.arch, args.dataset, None, device)
    strategies = build_strategies(blur_sigma=args.blur_sigma, noise_std=args.noise_std)
    perturb_fn = strategies[args.strategy]

    train_ds = get_dataset(args.dataset, args.data_root, bundle.transform, split="train")
    n = min(args.n_train_images, len(train_ds))
    indices = rng.choice(len(train_ds), size=n, replace=False).tolist()
    print(f"Fine-tuning {args.arch}/{args.dataset} against '{args.strategy}' "
          f"on {n} training images, {args.epochs} epochs")

    optimizer = torch.optim.AdamW(bundle.model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        rng.shuffle(indices)
        total_loss = 0.0
        for count, idx in enumerate(indices, start=1):
            x, _ = train_ds[idx]
            x = x.unsqueeze(0).to(device)
            H, W = x.shape[-2:]

            bundle.model.eval()
            with torch.no_grad():
                pseudo_label = bundle.forward_logits(x).argmax(1)

            mask = random_mask(H, W, rng)
            x_pert = perturb_fn(x, mask, bundle.mean, bundle.std)

            bundle.model.train()
            logits = bundle.forward_logits(x_pert)
            loss = F.cross_entropy(logits, pseudo_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if count % args.log_every == 0:
                print(f"  epoch {epoch+1} [{count}/{n}] avg_loss={total_loss/count:.4f}")

        print(f"epoch {epoch+1}/{args.epochs}: avg_loss={total_loss/n:.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.arch}_{args.dataset}_{args.strategy}_robust.pt"
    bundle.model.eval()
    torch.save(bundle.model.state_dict(), out_path)
    print(f"Saved robustified checkpoint to {out_path}")


if __name__ == "__main__":
    main()
