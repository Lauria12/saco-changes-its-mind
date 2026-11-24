"""Attention-weight capture, generalized across the three model families used here.

torchvision ViT (vit_b16, vit_l16) exposes nn.MultiheadAttention modules that only
return weights if asked at call time, so we monkey-patch .forward() per layer.
timm DeiT exposes post-softmax weights as the *input* to attn_drop, so we hook that.
HF transformers models (used for the CIFAR fine-tunes) return attentions directly
from the forward pass when output_attentions=True -- no hooking needed.
"""
import torch
import torch.nn.functional as F


def capture_attentions_torchvision(model, x, layer_indices):
    """Returns a list of [B, heads, N, N] attention tensors, one per requested layer."""
    captured = {}
    handles = []
    patched = []
    layers = model.encoder.layers

    for i in layer_indices:
        attn_module = layers[i].self_attention
        original_forward = attn_module.forward

        def make_forward(orig):
            def forward_with_weights(query, key, value, **kwargs):
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                return orig(query, key, value, **kwargs)
            return forward_with_weights

        def make_hook(idx):
            def hook_fn(module, inputs, output):
                if isinstance(output, tuple) and len(output) > 1:
                    captured[idx] = output[1].detach().cpu()
            return hook_fn

        attn_module.forward = make_forward(original_forward)
        handles.append(attn_module.register_forward_hook(make_hook(i)))
        patched.append((attn_module, original_forward))

    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for handle in handles:
            handle.remove()
        for attn_module, original_forward in patched:
            attn_module.forward = original_forward

    return [captured[i] for i in layer_indices if i in captured]


def capture_attentions_timm(model, x, layer_indices):
    """Requires block.attn.fused_attn = False to have been set at load time,
    otherwise attn_drop never receives a materialized attention matrix."""
    captured = {}
    handles = []
    blocks = model.blocks

    for i in layer_indices:
        def make_hook(idx):
            def hook(module, inp, out):
                if inp and inp[0] is not None:
                    captured[idx] = inp[0].detach().cpu()
            return hook
        handles.append(blocks[i].attn.attn_drop.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            _ = model(x)
    finally:
        for handle in handles:
            handle.remove()

    return [captured[i] for i in layer_indices if i in captured]


def capture_attentions_hf(model, x, layer_indices):
    with torch.no_grad():
        outputs = model(pixel_values=x, output_attentions=True)
    return [outputs.attentions[i].detach().cpu() for i in layer_indices]


_DISPATCH = {
    "torchvision_vit": capture_attentions_torchvision,
    "timm": capture_attentions_timm,
    "hf": capture_attentions_hf,
}


def capture_attentions(model, x, family, layer_indices):
    return _DISPATCH[family](model, x, layer_indices)


def capture_attentions_with_grad_torchvision(model, x, layer_indices):
    """Like capture_attentions_torchvision, but keeps attention tensors attached to
    the autograd graph and returns the logits from the SAME forward pass. Callers
    must backward() from those logits, not a separately-computed forward_logits(x)
    call -- a second call builds a disconnected graph and every .grad silently
    stays None.

    This does NOT simply hook nn.MultiheadAttention's need_weights=True output like
    the no-grad version above. Verified empirically: that path returns a
    numerically-correct but graph-DISCONNECTED weights tensor -- attn_output is
    computed via an internal fused kernel that never actually consumes the returned
    weights tensor, so retain_grad() on it never populates .grad, even though the
    tensor itself has requires_grad=True and a real grad_fn (it's just not an
    ancestor of the logits). Fix: manually reimplement the per-head
    QKV -> scaled-dot-product -> softmax -> matmul computation using the module's
    own in_proj_weight/in_proj_bias/out_proj, so the captured weights tensor is a
    genuine ancestor of attn_output. Verified numerically faithful to the original
    module's output (max abs diff ~1e-7 against the unmodified forward pass)."""
    captured = {}
    patched = []
    layers = model.encoder.layers

    for i in layer_indices:
        mha = layers[i].self_attention
        assert mha.batch_first, "assumes batch_first=True, as torchvision's ViT uses"
        original_forward = mha.forward
        num_heads = mha.num_heads
        head_dim = mha.embed_dim // num_heads
        scale = head_dim ** -0.5
        w_q, w_k, w_v = mha.in_proj_weight.chunk(3)
        if mha.in_proj_bias is not None:
            b_q, b_k, b_v = mha.in_proj_bias.chunk(3)
        else:
            b_q = b_k = b_v = None

        def make_forward(idx, w_q=w_q, w_k=w_k, w_v=w_v, b_q=b_q, b_k=b_k, b_v=b_v,
                          num_heads=num_heads, head_dim=head_dim, scale=scale, mha=mha):
            def custom_forward(query, key, value, **kwargs):
                B, N, E = query.shape
                Q = F.linear(query, w_q, b_q)
                K = F.linear(key, w_k, b_k)
                V = F.linear(value, w_v, b_v)

                def split_heads(t):
                    return t.view(B, N, num_heads, head_dim).transpose(1, 2)

                Qh, Kh, Vh = split_heads(Q), split_heads(K), split_heads(V)
                scores = torch.matmul(Qh, Kh.transpose(-2, -1)) * scale
                weights = torch.softmax(scores, dim=-1)  # eval mode: no attn dropout to apply
                weights.retain_grad()
                captured[idx] = weights

                out = torch.matmul(weights, Vh)
                out = out.transpose(1, 2).reshape(B, N, E)
                out = mha.out_proj(out)
                return out, weights
            return custom_forward

        mha.forward = make_forward(i)
        patched.append((mha, original_forward))

    try:
        with torch.enable_grad():
            logits = model(x)
    finally:
        for mha, original_forward in patched:
            mha.forward = original_forward

    return [captured[i] for i in layer_indices if i in captured], logits


def capture_attentions_with_grad_timm(model, x, layer_indices):
    """Grad-attached counterpart to capture_attentions_timm. Still requires
    block.attn.fused_attn = False (set at load time), otherwise attn_drop never
    receives a materialized attention matrix to hook."""
    captured = {}
    handles = []
    blocks = model.blocks

    for i in layer_indices:
        def make_hook(idx):
            def hook(module, inp, out):
                if inp and inp[0] is not None:
                    attn = inp[0]
                    attn.retain_grad()
                    captured[idx] = attn
            return hook
        handles.append(blocks[i].attn.attn_drop.register_forward_hook(make_hook(i)))

    try:
        with torch.enable_grad():
            logits = model(x)
    finally:
        for handle in handles:
            handle.remove()

    return [captured[i] for i in layer_indices if i in captured], logits


def capture_attentions_with_grad_hf(model, x, layer_indices):
    """Grad-attached counterpart to capture_attentions_hf. Requires
    attn_implementation="eager" (already forced at load time in
    saco/models.py::_load_hf_cifar), otherwise output_attentions=True returns
    nothing to retain_grad() on."""
    with torch.enable_grad():
        outputs = model(pixel_values=x, output_attentions=True)
    attns = [outputs.attentions[i] for i in layer_indices]
    for a in attns:
        a.retain_grad()
    return attns, outputs.logits


_DISPATCH_GRAD = {
    "torchvision_vit": capture_attentions_with_grad_torchvision,
    "timm": capture_attentions_with_grad_timm,
    "hf": capture_attentions_with_grad_hf,
}


def capture_attentions_with_grad(model, x, family, layer_indices):
    return _DISPATCH_GRAD[family](model, x, layer_indices)
