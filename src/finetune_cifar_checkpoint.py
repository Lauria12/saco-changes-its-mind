#!/usr/bin/env python
"""Fine-tunes an HF-format ImageNet-pretrained checkpoint on CIFAR-10/100, for the
3 (arch, dataset) combos with no trustworthy checkpoint on the Hub -- see
configs/cifar_checkpoints.py for the registry and CIFAR_FINETUNE_BASE_CHECKPOINTS
for which base checkpoint each arch fine-tunes from.

Full fine-tune (not linear probe), matching the quality bar of the existing
community CIFAR checkpoints already in the registry. Trains on 95% of the CIFAR
train split, holds out 5% for model selection, and NEVER touches the official test
split (that's what run_experiment.py evaluates SaCo faithfulness against later, via
saco.data.get_dataset(..., split="val")).

Example:
    python finetune_cifar_checkpoint.py --arch vit_l16 --dataset cifar10 \\
        --data-root ../data --output-dir ../checkpoints --epochs 5
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    Trainer, TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from saco.data import get_dataset
from configs.cifar_checkpoints import CIFAR_FINETUNE_BASE_CHECKPOINTS

NUM_CLASSES = {"cifar10": 10, "cifar100": 100}


class HFCifarDataset(Dataset):
    """Wraps a raw torchvision CIFAR dataset (PIL images, int targets) for HF
    Trainer: applies light train-time augmentation, then routes through the base
    checkpoint's own AutoImageProcessor -- the exact same call _load_hf_cifar()
    uses at inference time in saco/models.py, so train/inference preprocessing
    match exactly."""

    def __init__(self, base_dataset, processor, indices, augment):
        self.base = base_dataset
        self.processor = processor
        self.indices = indices
        self.augment = transforms.Compose([
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]) if augment else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.augment is not None:
            img = self.augment(img)
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "labels": int(label)}


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float((preds == labels).mean())}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, choices=list(CIFAR_FINETUNE_BASE_CHECKPOINTS))
    p.add_argument("--dataset", required=True, choices=list(NUM_CLASSES))
    p.add_argument("--data-root", default="./data")
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=None,
                    help="Default: 32 for vit_l16, 64 for deit_b16")
    p.add_argument("--grad-accum", type=int, default=None,
                    help="Default: 4 for vit_l16, 2 for deit_b16 (effective batch 128)")
    p.add_argument("--lr", type=float, default=None,
                    help="Default: 2e-5 for vit_l16, 3e-5 for deit_b16")
    p.add_argument("--holdout-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    defaults = {
        "vit_l16": {"batch_size": 32, "grad_accum": 4, "lr": 2e-5},
        "deit_b16": {"batch_size": 64, "grad_accum": 2, "lr": 3e-5},
    }[args.arch]
    batch_size = args.batch_size or defaults["batch_size"]
    grad_accum = args.grad_accum or defaults["grad_accum"]
    lr = args.lr or defaults["lr"]

    base_repo = CIFAR_FINETUNE_BASE_CHECKPOINTS[args.arch]
    num_labels = NUM_CLASSES[args.dataset]

    train_full = get_dataset(args.dataset, args.data_root, transform=None, split="train")
    classes = train_full.classes
    id2label = {i: name for i, name in enumerate(classes)}
    label2id = {name: i for i, name in id2label.items()}

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(train_full))
    n_holdout = int(len(train_full) * args.holdout_frac)
    holdout_idx, train_idx = perm[:n_holdout].tolist(), perm[n_holdout:].tolist()

    print(f"Base: {base_repo} | arch={args.arch} dataset={args.dataset} "
          f"num_labels={num_labels} | train={len(train_idx)} holdout={len(holdout_idx)}")

    processor = AutoImageProcessor.from_pretrained(base_repo)
    model = AutoModelForImageClassification.from_pretrained(
        base_repo, num_labels=num_labels, id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True, attn_implementation="eager",
    )

    train_ds = HFCifarDataset(train_full, processor, train_idx, augment=True)
    holdout_ds = HFCifarDataset(train_full, processor, holdout_idx, augment=False)

    ckpt_dir = Path(args.output_dir) / f"{args.arch}_{args.dataset}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(ckpt_dir / "_trainer_state"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=8,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=holdout_ds,
        compute_metrics=compute_metrics,
    )

    resume = get_last_checkpoint(training_args.output_dir)
    if resume:
        print(f"Resuming from {resume}")
    trainer.train(resume_from_checkpoint=resume)

    holdout_metrics = trainer.evaluate()
    print(f"Held-out accuracy (model selection only, not the official test set): "
          f"{holdout_metrics['eval_accuracy']:.4f}")

    trainer.save_model(str(ckpt_dir))
    processor.save_pretrained(str(ckpt_dir))

    # Official CIFAR test set (== saco.data.get_dataset(..., split="val")) -- report-only,
    # never used for model selection or early stopping.
    test_full = get_dataset(args.dataset, args.data_root, transform=None, split="val")
    test_ds = HFCifarDataset(test_full, processor, list(range(len(test_full))), augment=False)
    test_metrics = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    print(f"Official test-set accuracy (report-only): {test_metrics['test_accuracy']:.4f}")

    info = {
        "arch": args.arch, "dataset": args.dataset, "base_checkpoint": base_repo,
        "epochs": args.epochs, "batch_size": batch_size, "grad_accum": grad_accum,
        "effective_batch_size": batch_size * grad_accum, "lr": lr, "seed": args.seed,
        "n_train": len(train_idx), "n_holdout": len(holdout_idx),
        "holdout_accuracy": holdout_metrics["eval_accuracy"],
        "test_accuracy": test_metrics["test_accuracy"],
    }
    with open(ckpt_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"Saved checkpoint + training_info.json to {ckpt_dir}")


if __name__ == "__main__":
    main()
