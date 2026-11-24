"""Registry of Hugging Face Hub checkpoints for CIFAR-10/CIFAR-100 fine-tunes.

The ImageNet-pretrained ViT-B/16, ViT-L/16, and DeiT-B/16 checkpoints (used for the
imagenet/imagenette runs) output 1000 ImageNet classes and cannot be pointed at
CIFAR-10 (10 classes) or CIFAR-100 (100 classes) directly. Each entry below must be a
real, verified repo ID of a model actually fine-tuned on that dataset.

Verified 2026-08-07 via Hub search (see chat history for the full trust breakdown):

- vit_b16/cifar10, vit_b16/cifar100: solid, well-documented community checkpoints.
- vit_l16/cifar10: NOT FOUND. No credible ViT-L/16 CIFAR-10 checkpoint exists on the
  Hub -- people fine-tune ViT-B on CIFAR in practice since 32x32 images give ViT-L no
  real benefit and it overfits faster on only 50k images. Fine-tune your own or drop
  ViT-L from the CIFAR-10 comparison.
- vit_l16/cifar100: only one candidate found (jialicheng/cifar100_vit-large), 93.09%
  reported accuracy but low trust signal (65 downloads/mo, 1 like, incomplete model
  card). Left populated below since it's the only option, but verify it yourself
  (spot-check predictions) before trusting it for a paper.
- deit_b16/cifar10, deit_b16/cifar100: NOT FOUND. Every candidate on the Hub either
  has no model card / unverifiable provenance (tzhao3/DeiT-CIFAR10,
  fancifulcrow/deit-base-patch16-224-finetuned-cifar10, ranksu/cifar10_deit_finetuning,
  tzhao3/DeiT-CIFAR100, ranksu/cifar100_deit_finetuning) or is a different architecture
  entirely (FALcon6/LGViT-DeiT-Cifar100 is a custom early-exit model, not a plain
  DeiT classifier). You'll need to fine-tune facebook/deit-base-patch16-224 on CIFAR
  yourself for these two combos.

An empty entry means "not yet found/verified"; run_experiment.py will raise a clear
error rather than silently falling back to something wrong, so leaving these empty is
safe -- it just blocks that specific arch/dataset combo until filled in.

2026-08-14: started fine-tuning the 3 missing combos ourselves (finetune_cifar_checkpoint.py
+ jobs/finetune_checkpoint.job) rather than waiting for a trustworthy Hub checkpoint to
appear. Base checkpoints used (see CIFAR_FINETUNE_BASE_CHECKPOINTS below) -- NOT the same
pretrained source as the torchvision/timm weights used for imagenet/imagenette elsewhere in
this pipeline, so cross-dataset faithfulness comparisons for vit_l16 in particular should be
read as "same architecture family," not "identical weights" (vit_b16's two existing Hub
checkpoints already have this same property).
Resulting checkpoints save to checkpoints/<arch>_<dataset>/ and are picked up
automatically below (_local_or_none) with zero code changes needed elsewhere once they exist.
"""
from pathlib import Path

CIFAR_FINETUNE_BASE_CHECKPOINTS = {
    # HF-format ImageNet-pretrained checkpoints to fine-tune from. NOT the torchvision
    # (vit_l16) / timm (deit_b16) weights used for the imagenet/imagenette evaluation --
    # those aren't in transformers/AutoModelForImageClassification format.
    "vit_l16": "google/vit-large-patch16-224",  # ViTConfig: 24 layers, 1024 hidden, 16 heads
    "deit_b16": "facebook/deit-base-patch16-224",  # NOT -distilled-: that has 2 special
    # tokens (CLS + distillation), and saco/explanations.py::_cls_to_patch_grid assumes
    # exactly 1 (drops 1 token, reshapes the rest into a square grid -- 2 tokens breaks
    # the square-grid math). Confirmed single-CLS-token ViTConfig via AutoConfig.
}

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # repository root
_CKPT_ROOT = _REPO_ROOT / "checkpoints"


def _local_or_none(arch: str, dataset: str):
    path = _CKPT_ROOT / f"{arch}_{dataset}"
    return str(path) if path.is_dir() else None


CIFAR_CHECKPOINTS = {
    "vit_b16": {
        "cifar10": "aaraki/vit-base-patch16-224-in21k-finetuned-cifar10",
        "cifar100": "Ahmed9275/Vit-Cifar100",
    },
    "vit_l16": {
        "cifar10": _local_or_none("vit_l16", "cifar10"),  # None until fine-tuned locally
        "cifar100": "jialicheng/cifar100_vit-large",  # low-trust, spot-check before relying on it
    },
    "deit_b16": {
        "cifar10": _local_or_none("deit_b16", "cifar10"),  # None until fine-tuned locally
        "cifar100": _local_or_none("deit_b16", "cifar100"),  # None until fine-tuned locally
    },
}
