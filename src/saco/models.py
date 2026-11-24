"""Model loading for the three architectures (ViT-B/16, ViT-L/16, DeiT-B/16) across
the four dataset targets (imagenet, imagenette, cifar10, cifar100).

ImageNet/Imagenette use the standard ImageNet-1k pretrained checkpoints shipped by
torchvision/timm. CIFAR-10/100 need checkpoints actually fine-tuned on those classes
-- see configs/cifar_checkpoints.py for the registry of which HF Hub repo to pull.
"""
import dataclasses
from typing import Callable, List, Optional

import torch
import torch.nn as nn

from . import attention as attn_mod

try:
    from configs.cifar_checkpoints import CIFAR_CHECKPOINTS
except ImportError:
    CIFAR_CHECKPOINTS = {}

ARCHES = ("vit_b16", "vit_l16", "deit_b16")
IMAGENET_DATASETS = ("imagenet", "imagenette")
CIFAR_DATASETS = ("cifar10", "cifar100")


@dataclasses.dataclass
class ModelBundle:
    model: nn.Module
    family: str  # "torchvision_vit" | "timm" | "hf"
    transform: Callable
    num_layers: int
    mean: List[float]
    std: List[float]
    num_classes: int

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.family == "hf":
            return self.model(pixel_values=x).logits
        return self.model(x)

    def get_attentions(self, x: torch.Tensor, layer_indices: List[int]):
        return attn_mod.capture_attentions(self.model, x, self.family, layer_indices)

    def get_attentions_with_grad(self, x: torch.Tensor, layer_indices: List[int]):
        """Like get_attentions, but returns (attn_list, logits) with attention
        tensors left attached to the autograd graph (retain_grad'd) instead of
        detached -- for methods (e.g. TransMM) that need gradients w.r.t. attention
        weights. logits come from the same forward pass as attn_list; backward()
        from them, not from a separate forward_logits(x) call."""
        return attn_mod.capture_attentions_with_grad(self.model, x, self.family, layer_indices)

    def logits_module(self) -> nn.Module:
        """A plain nn.Module wrapper so Captum (which calls model(x) and expects a
        tensor) works uniformly across all three families."""
        bundle = self

        class _Wrapper(nn.Module):
            def forward(self, x):
                return bundle.forward_logits(x)

        return _Wrapper()


def _load_torchvision_vit(arch: str, device, checkpoint: Optional[str] = None):
    from torchvision.models import (
        vit_b_16, ViT_B_16_Weights,
        vit_l_16, ViT_L_16_Weights,
    )
    if arch == "vit_b16":
        weights = ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)
    else:
        weights = ViT_L_16_Weights.DEFAULT
        model = vit_l_16(weights=weights)
    if checkpoint:
        # Local state-dict override (e.g. a perturbation-robustness fine-tune) --
        # loaded on top of the pretrained weights above, same architecture/head,
        # just different learned parameters.
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    model = model.eval().to(device)
    transform = weights.transforms()
    mean, std = list(transform.mean), list(transform.std)
    num_layers = len(model.encoder.layers)
    num_classes = len(weights.meta["categories"])
    return ModelBundle(model, "torchvision_vit", transform, num_layers, mean, std, num_classes)


def _load_timm_deit(device, checkpoint: Optional[str] = None):
    import timm
    from timm.data import resolve_model_data_config, create_transform

    model = timm.create_model("deit_base_patch16_224", pretrained=True).eval().to(device)
    if checkpoint:
        state_dict = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False

    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    mean, std = list(data_config["mean"]), list(data_config["std"])
    num_layers = len(model.blocks)
    num_classes = model.num_classes
    return ModelBundle(model, "timm", transform, num_layers, mean, std, num_classes)


def _load_hf_cifar(arch: str, dataset: str, checkpoint: Optional[str], device):
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    state_dict_override = None
    if checkpoint and checkpoint.endswith(".pt"):
        # A raw state-dict override (e.g. a perturbation-robustness fine-tune saved
        # via torch.save(model.state_dict(), ...) in finetune_perturbation_robust.py),
        # not an HF Hub repo id or a from_pretrained-compatible local directory.
        # Resolve the base registry checkpoint below as usual, then apply this state
        # dict on top of it.
        state_dict_override = checkpoint
        checkpoint = None

    repo_id = checkpoint or CIFAR_CHECKPOINTS.get(arch, {}).get(dataset)
    if not repo_id:
        raise ValueError(
            f"No HF Hub checkpoint registered for arch={arch!r} dataset={dataset!r}. "
            f"Pass --checkpoint <hf-repo-id> explicitly, or add an entry to "
            f"configs/cifar_checkpoints.py."
        )

    processor = AutoImageProcessor.from_pretrained(repo_id)
    # attn_implementation="eager" is required: transformers' default "sdpa" backend
    # doesn't materialize attention weights, so output_attentions=True is silently
    # ignored and outputs.attentions comes back empty -- breaks RawAtt/Rollout.
    model = AutoModelForImageClassification.from_pretrained(
        repo_id, attn_implementation="eager"
    )
    if state_dict_override:
        model.load_state_dict(torch.load(state_dict_override, map_location=device))
    model = model.eval().to(device)

    mean, std = processor.image_mean, processor.image_std

    def transform(img):
        # Route through the checkpoint's own processor rather than reimplementing
        # its resize/rescale/normalize logic, so preprocessing exactly matches
        # what the model was fine-tuned with.
        return processor(images=img, return_tensors="pt")["pixel_values"][0]

    num_layers = model.config.num_hidden_layers
    num_classes = model.config.num_labels
    return ModelBundle(model, "hf", transform, num_layers, mean, std, num_classes)


def load_model(arch: str, dataset: str, checkpoint: Optional[str], device) -> ModelBundle:
    if arch not in ARCHES:
        raise ValueError(f"Unknown arch {arch!r}, expected one of {ARCHES}")

    if dataset in IMAGENET_DATASETS:
        if arch in ("vit_b16", "vit_l16"):
            return _load_torchvision_vit(arch, device, checkpoint)
        return _load_timm_deit(device, checkpoint)

    if dataset in CIFAR_DATASETS:
        return _load_hf_cifar(arch, dataset, checkpoint, device)

    raise ValueError(f"Unknown dataset {dataset!r}")
