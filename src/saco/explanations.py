"""The four explanation methods from Results.md, generalized over ModelBundle so
the same code runs against torchvision ViT, timm DeiT, or HF transformers models.
"""
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients


def random_explanation(bundle, x, target_class=None):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    return torch.rand(x.shape[-2], x.shape[-1])


def ig_explanation(bundle, x, target_class, n_steps=20, internal_batch_size=None):
    if x.ndim == 3:
        x = x.unsqueeze(0)
    ig = IntegratedGradients(bundle.logits_module())
    attr = ig.attribute(x, baselines=torch.zeros_like(x),
                         target=target_class, n_steps=n_steps,
                         internal_batch_size=internal_batch_size)
    return attr.abs().mean(dim=1).squeeze(0)


def _cls_to_patch_grid(cls_to_patches, out_size):
    grid = int(cls_to_patches.numel() ** 0.5)
    sal = cls_to_patches.reshape(1, 1, grid, grid)
    sal = F.interpolate(sal, size=out_size, mode="bilinear", align_corners=False)
    sal = sal.squeeze().float()
    return (sal - sal.min()) / (sal.max() + 1e-8)


def raw_attention_explanation(bundle, x, target_class=None):
    """CLS-to-patch attention from the last block, averaged over heads."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    attns = bundle.get_attentions(x, [bundle.num_layers - 1])
    if not attns:
        return torch.rand(x.shape[-2], x.shape[-1])
    attn = attns[0][0].mean(0)          # [N, N], heads averaged
    cls_to_patches = attn[0, 1:]        # CLS row, drop CLS column
    return _cls_to_patch_grid(cls_to_patches, x.shape[-2:])


def rollout_explanation(bundle, x, target_class=None):
    """Attention Rollout across all blocks (Abnar & Zuidema, 2020)."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    attns = bundle.get_attentions(x, list(range(bundle.num_layers)))
    if not attns:
        return torch.rand(x.shape[-2], x.shape[-1])

    N = attns[0].shape[-1]
    result = torch.eye(N)
    for attn in attns:
        attn_mean = attn[0].mean(0)
        attn_res = (attn_mean + torch.eye(N)) / 2.0
        attn_res = attn_res / attn_res.sum(dim=-1, keepdim=True)
        result = torch.matmul(attn_res, result)

    cls_attn = result[0, 1:]
    return _cls_to_patch_grid(cls_attn, x.shape[-2:])


def transmm_explanation(bundle, x, target_class):
    """Transformer-MM (Chefer, Gur & Wolf, ICCV 2021, DOI 10.1109/ICCV48922.2021.00045)
    -- gradient-weighted Rollout. Per layer: cam = ReLU(grad * attn), averaged over
    heads; aggregate additively R <- R + cam @ R starting from R = I. Deliberately
    NOT the same recursion as rollout_explanation above (that one is multiplicative,
    (attn+I)/2 row-normalized) -- the two are expected to agree only qualitatively,
    not numerically."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    layer_indices = list(range(bundle.num_layers))
    attns, logits = bundle.get_attentions_with_grad(x, layer_indices)
    if not attns:
        return torch.rand(x.shape[-2], x.shape[-1])

    bundle.model.zero_grad(set_to_none=True)
    logits[0, target_class].backward()

    N = attns[0].shape[-1]
    device = attns[0].device
    with torch.no_grad():
        R = torch.eye(N, device=device)
        for attn in attns:
            grad = attn.grad
            if grad is None:
                raise RuntimeError(
                    "TransMM: attention tensor has no grad -- logits and attentions "
                    "must come from the same forward pass (see get_attentions_with_grad)."
                )
            cam = (grad[0] * attn[0]).clamp(min=0).mean(dim=0)  # [N, N], heads averaged
            R = R + torch.matmul(cam, R)

    cls_to_patches = R[0, 1:]
    return _cls_to_patch_grid(cls_to_patches, x.shape[-2:])


METHODS = {
    "Random": random_explanation,
    "IG": ig_explanation,
    "RawAtt": raw_attention_explanation,
    "Rollout": rollout_explanation,
    "TransMM": transmm_explanation,
}
