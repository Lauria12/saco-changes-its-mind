"""The SaCo rank-bin metric core -- direct port of Algorithm 1 from the paper, as
implemented in saco_deit_analysis.ipynb. Model-agnostic: operates on saliency maps
and confidence scalars only.
"""
import numpy as np
import torch
import torch.nn.functional as F


def normalize_saliency(sal, H, W):
    if isinstance(sal, torch.Tensor):
        sal = sal.detach().float().cpu()
    else:
        sal = torch.tensor(np.asarray(sal), dtype=torch.float32)
    sal = sal.squeeze()
    if sal.ndim == 3:
        sal = sal.mean(0)
    if sal.shape != (H, W):
        sal = F.interpolate(sal.unsqueeze(0).unsqueeze(0), (H, W),
                             mode="bilinear", align_corners=False).squeeze()
    sal = sal.abs()
    sal = sal - sal.min()
    if sal.max() > 0:
        sal = sal / sal.max()
    return sal.numpy()


def make_rank_masks(norm_sal, K=10):
    H, W = norm_sal.shape
    total = H * W
    group_size = total // K
    order = np.argsort(norm_sal.flatten())
    masks = []
    for k in range(K):
        start = k * group_size
        end = (k + 1) * group_size if k < K - 1 else total
        m = np.zeros(total, dtype=bool)
        m[order[start:end]] = True
        masks.append(m.reshape(H, W))
    return masks


def saco_coefficient(sal_sums, conf_drops):
    """Algorithm 1 from the paper. Range [-1, 1], higher = more faithful."""
    K = len(sal_sums)
    f_val = 0.0
    total = 0.0
    for i in range(K - 1):
        for j in range(i + 1, K):
            si, sj = sal_sums[i], sal_sums[j]
            di, dj = conf_drops[i], conf_drops[j]
            w = abs(si - sj)
            total += w
            if si >= sj:
                f_val += w if di >= dj else -w
            else:
                f_val += w if dj >= di else -w
    return f_val / total if total > 0 else 0.0


def analyse_image(bundle, method_fn, x, strategies, K=10):
    """Runs one image through one explanation method and all perturbation
    strategies, returning the SaCo coefficient per strategy."""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    device = next(bundle.model.parameters()).device
    x = x.to(device)
    _, _, H, W = x.shape

    with torch.no_grad():
        probs = torch.softmax(bundle.forward_logits(x), dim=1)
    target_class = int(probs.argmax(1).item())
    base_conf = float(probs[0, target_class].item())

    raw_sal = method_fn(bundle, x, target_class)
    norm_sal = normalize_saliency(raw_sal, H, W)
    masks = make_rank_masks(norm_sal, K=K)
    sal_sums = [float(norm_sal[m].sum()) for m in masks]

    scores = {}
    for strat_name, perturb_fn in strategies.items():
        drops = []
        for mask in masks:
            x_pert = perturb_fn(x, mask, bundle.mean, bundle.std)
            with torch.no_grad():
                pert_probs = torch.softmax(bundle.forward_logits(x_pert), dim=1)
            pert_conf = float(pert_probs[0, target_class].item())
            drops.append(base_conf - pert_conf)
        scores[strat_name] = saco_coefficient(sal_sums, drops)

    return {"target_class": target_class, "base_conf": base_conf, "scores": scores}
