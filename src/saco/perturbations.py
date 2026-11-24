"""The five perturbation strategies from Results.md, generalized to take the
normalization mean/std actually used by the model's preprocessing transform.

The original notebook hardcoded ImageNet mean/std in perturb_inpaint (its only
strategy that needs to round-trip through raw [0,1] pixel space for OpenCV). That
was correct there because everything was ImageNet-pretrained, but would silently
corrupt CIFAR runs, which use different normalization stats.
"""
import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def perturb_mean(x, mask, mean=None, std=None):
    x_pert = x.clone()
    C = x_pert.shape[1]
    idx = np.flatnonzero(mask.reshape(-1))
    if len(idx) == 0:
        return x_pert
    mean_val = x_pert.mean(dim=(2, 3), keepdim=True)
    flat = x_pert.view(1, C, -1)
    flat[:, :, idx] = mean_val.view(1, C, 1).expand(1, C, len(idx))
    return flat.view_as(x_pert)


def perturb_blur(x, mask, mean=None, std=None, sigma=10):
    x_pert = x.clone()
    C = x_pert.shape[1]
    idx = np.flatnonzero(mask.reshape(-1))
    if len(idx) == 0:
        return x_pert
    img_np = x_pert[0].cpu().numpy()
    blurred = np.stack([gaussian_filter(img_np[c], sigma=sigma) for c in range(C)])
    blurred = torch.tensor(blurred, dtype=x_pert.dtype, device=x_pert.device)
    flat = x_pert.view(1, C, -1)
    flat[:, :, idx] = blurred.view(C, -1)[:, idx].unsqueeze(0)
    return flat.view_as(x_pert)


def perturb_noise(x, mask, mean=None, std=None, noise_std=0.3):
    x_pert = x.clone()
    C = x_pert.shape[1]
    idx = np.flatnonzero(mask.reshape(-1))
    if len(idx) == 0:
        return x_pert
    flat = x_pert.view(1, C, -1)
    flat[:, :, idx] = torch.randn(1, C, len(idx), device=x_pert.device) * noise_std
    return flat.view_as(x_pert)


def perturb_shuffle(x, mask, mean=None, std=None):
    x_pert = x.clone()
    C = x_pert.shape[1]
    idx = np.flatnonzero(mask.reshape(-1))
    if len(idx) < 2:
        return x_pert
    perm = np.random.permutation(len(idx))
    flat = x_pert.view(1, C, -1)
    flat[:, :, idx] = flat[:, :, idx[perm]].clone()
    return flat.view_as(x_pert)


def perturb_inpaint(x, mask, mean, std):
    x_pert = x.clone()
    mean_arr = np.asarray(mean, dtype=np.float32)
    std_arr = np.asarray(std, dtype=np.float32)

    img_np = x_pert[0].permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np * std_arr + mean_arr, 0, 1)
    img_u8 = (img_np * 255).astype(np.uint8)
    mask_u8 = mask.astype(np.uint8) * 255

    inpainted = cv2.inpaint(img_u8, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    inpainted_f = inpainted.astype(np.float32) / 255.0
    inpainted_n = (inpainted_f - mean_arr) / std_arr
    x_pert[0] = torch.tensor(inpainted_n, dtype=x_pert.dtype, device=x_pert.device).permute(2, 0, 1)
    return x_pert


def build_strategies(blur_sigma=10, noise_std=0.3):
    return {
        "mean": lambda x, mask, mean, std: perturb_mean(x, mask, mean, std),
        "blur": lambda x, mask, mean, std: perturb_blur(x, mask, mean, std, sigma=blur_sigma),
        "noise": lambda x, mask, mean, std: perturb_noise(x, mask, mean, std, noise_std=noise_std),
        "shuffle": lambda x, mask, mean, std: perturb_shuffle(x, mask, mean, std),
        "inpaint": lambda x, mask, mean, std: perturb_inpaint(x, mask, mean, std),
    }
