import math
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv


def _ensure_batched_rgb(rgb: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if rgb.ndim == 3:
        if rgb.shape[0] != 3:
            raise ValueError(f"Expected RGB tensor with shape [3,H,W], got {tuple(rgb.shape)}")
        return rgb.unsqueeze(0), True
    if rgb.ndim == 4:
        if rgb.shape[1] != 3:
            raise ValueError(f"Expected RGB tensor with shape [B,3,H,W], got {tuple(rgb.shape)}")
        return rgb, False
    raise ValueError(f"Expected RGB tensor with 3 or 4 dims, got {rgb.ndim}")


def _ensure_batched_maps(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.ndim == 2:
        return x.unsqueeze(0), True
    if x.ndim == 3:
        return x, False
    raise ValueError(f"Expected map tensor with shape [H,W] or [B,H,W], got {tuple(x.shape)}")


def _ensure_batched_bins(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim == 4:
        return x, False
    raise ValueError(
        f"Expected bins tensor with shape [K,H,W] or [B,K,H,W] (or [B,N*K,H,W]), got {tuple(x.shape)}"
    )


def rgb_tensor_to_hsv(rgb: torch.Tensor) -> torch.Tensor:
    rgb_batched, squeeze = _ensure_batched_rgb(rgb)
    rgb_np = rgb_batched.detach().cpu().permute(0, 2, 3, 1).numpy()
    rgb_np = np.clip(rgb_np, 0.0, 1.0)
    hsv_np = np.stack([rgb_to_hsv(img) for img in rgb_np], axis=0)
    hsv = torch.from_numpy(hsv_np).to(device=rgb_batched.device, dtype=rgb_batched.dtype)
    hsv = hsv.permute(0, 3, 1, 2).contiguous()
    return hsv[0] if squeeze else hsv


def hsv_tensor_to_rgb(hsv: torch.Tensor) -> torch.Tensor:
    hsv_batched, squeeze = _ensure_batched_rgb(hsv)
    hsv_np = hsv_batched.detach().cpu().permute(0, 2, 3, 1).numpy()
    hsv_np = np.clip(hsv_np, 0.0, 1.0)
    rgb_np = np.stack([hsv_to_rgb(img) for img in hsv_np], axis=0)
    rgb = torch.from_numpy(rgb_np).to(device=hsv_batched.device, dtype=hsv_batched.dtype)
    rgb = rgb.permute(0, 3, 1, 2).contiguous()
    return rgb[0] if squeeze else rgb


def _normalize_step(step: int, order: int) -> int:
    if order <= 0:
        return int(step)
    return int(step) % int(order)


def build_group_elements(r2_act, hue_step: int = 0, rot_step: int = 0) -> Dict:
    r"""
    Build consistent group elements for rotHueOnR2.

    For N=1 the fiber group collapses to C_H and hue elements are G.element(k).
    For N>1 the fiber group is C_N x C_H and elements are built via inclusion maps.
    """
    G = r2_act.fibergroup
    is_direct_product = hasattr(G, "G1") and hasattr(G, "G2") and hasattr(G, "inclusion1")

    if is_direct_product:
        rot_order = G.G1.order()
        hue_order = G.G2.order()
        rot_step_mod = _normalize_step(rot_step, rot_order)
        hue_step_mod = _normalize_step(hue_step, hue_order)

        g_rot = G.inclusion1(G.G1.element(rot_step_mod))
        g_hue = G.inclusion2(G.G2.element(hue_step_mod))
        identity = G.identity

        # Explicit identity check for direct-product case.
        assert G.inclusion1(G.G1.identity) @ G.inclusion2(G.G2.identity) == identity

        g = g_rot @ g_hue
        return {
            "group": G,
            "identity": identity,
            "rotation": g_rot,
            "hue": g_hue,
            "combined": g,
            "rot_order": rot_order,
            "hue_order": hue_order,
            "rot_step": rot_step_mod,
            "hue_step": hue_step_mod,
            "is_hue_only": False,
        }

    # N=1 case: subgroup reduction returns C_H directly.
    hue_order = G.order()
    hue_step_mod = _normalize_step(hue_step, hue_order)
    identity = G.identity
    g_hue = G.element(hue_step_mod)

    # Explicit identity check for cyclic case.
    assert G.element(0) == identity

    return {
        "group": G,
        "identity": identity,
        "rotation": identity,
        "hue": g_hue,
        "combined": g_hue,
        "rot_order": 1,
        "hue_order": hue_order,
        "rot_step": 0,
        "hue_step": hue_step_mod,
        "is_hue_only": True,
    }


def encode_rgb_to_huebins(rgb: torch.Tensor, K: int, mode: str = "soft") -> Dict[str, torch.Tensor]:
    r"""
    Convert RGB to hue bins while preserving saturation/value separately.

    mode="soft": 2-bin circular interpolation (sine weights), better reconstruction.
    mode="onehot": nearest-bin assignment.
    """
    if K <= 1:
        raise ValueError(f"K must be > 1, got {K}")
    if mode not in {"soft", "onehot"}:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'soft' or 'onehot'.")

    rgb_batched, squeeze = _ensure_batched_rgb(rgb)
    hsv = rgb_tensor_to_hsv(rgb_batched)
    hue = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]

    B, H, W = hue.shape
    bins = torch.zeros(B, K, H, W, device=rgb_batched.device, dtype=rgb_batched.dtype)

    if mode == "onehot":
        idx = torch.floor(hue * K).long() % K
        bins.scatter_(1, idx.unsqueeze(1), 1.0)
    else:
        # Two-neighbor circular interpolation using sine weights.
        # This pairs naturally with circular-mean decoding.
        delta = 2.0 * math.pi / K
        theta = hue * (2.0 * math.pi)
        idx0 = torch.floor(theta / delta).long() % K
        idx1 = (idx0 + 1) % K
        phi = theta - idx0.to(theta.dtype) * delta

        w0 = torch.sin(delta - phi)
        w1 = torch.sin(phi)
        denom = (w0 + w1).clamp_min(1e-8)
        w0 = w0 / denom
        w1 = w1 / denom

        bins.scatter_add_(1, idx0.unsqueeze(1), w0.unsqueeze(1))
        bins.scatter_add_(1, idx1.unsqueeze(1), w1.unsqueeze(1))

    if squeeze:
        return {
            "bins": bins[0],
            "hue": hue[0],
            "sat": sat[0],
            "val": val[0],
        }
    return {
        "bins": bins,
        "hue": hue,
        "sat": sat,
        "val": val,
    }


def decode_huebins_to_rgb(
    bins: torch.Tensor,
    sat: torch.Tensor,
    val: torch.Tensor,
    K: int,
    mode: str = "circular_mean",
    normalize_v_for_display: bool = False,
    n_rotation_bins: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    r"""
    Decode hue bins into RGB using provided saturation/value maps.

    Supports bins of shape:
    - [K,H,W] or [B,K,H,W]
    - [N*K,H,W] or [B,N*K,H,W] (rotation+hue regular representation), collapsed over N.
    """
    if mode not in {"circular_mean", "argmax"}:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'circular_mean' or 'argmax'.")

    bins_batched, squeeze_bins = _ensure_batched_bins(bins)
    sat_batched, squeeze_sat = _ensure_batched_maps(sat)
    val_batched, squeeze_val = _ensure_batched_maps(val)

    if squeeze_sat != squeeze_val:
        raise ValueError("sat and val must both be batched or both be unbatched")

    B, C, H, W = bins_batched.shape
    if sat_batched.shape != (B, H, W) or val_batched.shape != (B, H, W):
        raise ValueError(
            f"Shape mismatch. bins={tuple(bins_batched.shape)}, sat={tuple(sat_batched.shape)}, val={tuple(val_batched.shape)}"
        )

    if n_rotation_bins is None:
        if C == K:
            n_rotation_bins = 1
        elif C % K == 0:
            n_rotation_bins = C // K
        else:
            raise ValueError(f"Cannot infer rotation bins from channels C={C} and K={K}")
    elif n_rotation_bins * K != C:
        raise ValueError(f"Expected C = n_rotation_bins * K, got C={C}, n_rotation_bins={n_rotation_bins}, K={K}")

    if n_rotation_bins > 1:
        hue_bins = bins_batched.view(B, n_rotation_bins, K, H, W).sum(dim=1)
    else:
        hue_bins = bins_batched

    if mode == "argmax":
        idx = hue_bins.argmax(dim=1).to(hue_bins.dtype)
        hue = idx / float(K)
    else:
        theta = torch.arange(K, device=hue_bins.device, dtype=hue_bins.dtype) * (2.0 * math.pi / K)
        cos_t = torch.cos(theta).view(1, K, 1, 1)
        sin_t = torch.sin(theta).view(1, K, 1, 1)

        x = (hue_bins * cos_t).sum(dim=1)
        y = (hue_bins * sin_t).sum(dim=1)
        hue = torch.atan2(y, x) / (2.0 * math.pi)
        hue = torch.remainder(hue, 1.0)

    if normalize_v_for_display:
        v_min = val_batched.amin(dim=(-2, -1), keepdim=True)
        v_max = val_batched.amax(dim=(-2, -1), keepdim=True)
        val_used = (val_batched - v_min) / (v_max - v_min + 1e-8)
    else:
        val_used = val_batched

    hsv = torch.stack((hue, sat_batched, val_used), dim=1)
    rgb = hsv_tensor_to_rgb(hsv)

    # If S/V were provided as single maps ([H,W]) and batch size is 1,
    # return single-image tensors for convenience in notebook usage.
    if squeeze_sat and B == 1:
        return {
            "rgb": rgb[0],
            "hue": hue[0],
            "hue_bins": hue_bins[0],
        }
    return {
        "rgb": rgb,
        "hue": hue,
        "hue_bins": hue_bins,
    }


def build_regular_hue_field(hue_bins: torch.Tensor, N: int) -> torch.Tensor:
    r"""
    Lift K hue bins into regular representation channels of C_N x C_K (size N*K).
    We place hue bins in rotation index 0 block.
    """
    if N <= 0:
        raise ValueError(f"N must be positive, got {N}")

    bins_batched, squeeze = _ensure_batched_bins(hue_bins)
    B, K, H, W = bins_batched.shape
    if N == 1:
        field = bins_batched
    else:
        field = torch.zeros(B, N * K, H, W, device=bins_batched.device, dtype=bins_batched.dtype)
        field[:, :K] = bins_batched
    return field[0] if squeeze else field


def make_hue_geometric_tensor(hue_bins: torch.Tensor, N: int, K: int):
    from escnn import gspaces
    import escnn.nn as enn

    r2_act = gspaces.rotHueOnR2(N=N, H=K)
    ft = enn.FieldType(r2_act, [r2_act.regular_repr])
    field = build_regular_hue_field(hue_bins, N)
    field_batched, _ = _ensure_batched_bins(field)
    gx = enn.GeometricTensor(field_batched, ft)
    return r2_act, ft, gx


def compute_mae(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x - y).abs().mean().item()


def compute_psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((max_val * max_val) / mse)


def assert_group_consistency(
    gx: Any,
    r2_act,
    hue_step: int = 1,
    rot_step: int = 1,
    atol: float = 1e-5,
    strict: bool = True,
) -> Dict[str, float]:
    r"""
    Validate identity and composition in representation space.

    For hue-only case (N=1), also compares transform(k) against channel roll and
    reports the sign convention detected once from data.
    """
    out: Dict[str, float] = {}

    elems = build_group_elements(r2_act, hue_step=hue_step, rot_step=rot_step)
    identity = elems["identity"]
    g = elems["combined"]

    id_err = (gx.transform(identity).tensor - gx.tensor).abs().max().item()
    out["identity_max_abs_err"] = id_err
    if strict and id_err > atol:
        raise AssertionError(f"Identity check failed: max abs err {id_err:.3e} > {atol:.3e}")

    if elems["rot_order"] > 1:
        g1 = build_group_elements(r2_act, hue_step=hue_step, rot_step=0)["combined"]
        g2 = build_group_elements(r2_act, hue_step=0, rot_step=rot_step)["combined"]
    else:
        g1 = build_group_elements(r2_act, hue_step=hue_step, rot_step=0)["combined"]
        g2 = build_group_elements(r2_act, hue_step=1, rot_step=0)["combined"]

    lhs = gx.transform(g1 @ g2).tensor
    rhs = gx.transform(g2).transform(g1).tensor
    comp_err = (lhs - rhs).abs().max().item()
    out["composition_max_abs_err"] = comp_err
    if strict and comp_err > atol:
        raise AssertionError(f"Composition check failed: max abs err {comp_err:.3e} > {atol:.3e}")

    if elems["is_hue_only"]:
        k = elems["hue_step"]
        transformed = gx.transform(g).tensor
        plus = torch.roll(gx.tensor, shifts=k, dims=1)
        minus = torch.roll(gx.tensor, shifts=-k, dims=1)
        plus_err = (transformed - plus).abs().max().item()
        minus_err = (transformed - minus).abs().max().item()
        if plus_err <= minus_err:
            roll_sign = 1
            roll_err = plus_err
        else:
            roll_sign = -1
            roll_err = minus_err

        out["roll_sign"] = float(roll_sign)
        out["roll_max_abs_err"] = roll_err
        if strict and roll_err > atol:
            raise AssertionError(f"Roll consistency failed: max abs err {roll_err:.3e} > {atol:.3e}")

    return out


def _plot_image_row(images: Sequence[np.ndarray], titles: Sequence[str], figsize: Tuple[int, int]):
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if len(images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0.0, 1.0))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    return fig, axes


def build_synthetic_hsv_image(
    height: int = 33,
    width: int = 33,
    saturation: float = 0.95,
    min_value: float = 0.25,
    max_value: float = 1.0,
) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, width).view(1, width).repeat(height, 1)
    y = torch.linspace(0.0, 1.0, height).view(height, 1).repeat(1, width)

    hue = x
    sat = torch.full_like(hue, saturation)
    val = min_value + (max_value - min_value) * y

    hsv = torch.stack((hue, sat, val), dim=0)
    return hsv_tensor_to_rgb(hsv)


def run_synthetic_hue_cycle_validation(
    N: int = 1,
    K: int = 4,
    hue_steps: Optional[Iterable[int]] = None,
    encoding_mode: str = "soft",
    decoding_mode: str = "circular_mean",
    strict_assertions: bool = True,
    show_plot: bool = True,
) -> Dict:
    if hue_steps is None:
        hue_steps = range(K)

    rgb = build_synthetic_hsv_image()
    enc_soft = encode_rgb_to_huebins(rgb, K, mode=encoding_mode)
    enc_onehot = encode_rgb_to_huebins(rgb, K, mode="onehot")

    r2_act, _, gx_soft = make_hue_geometric_tensor(enc_soft["bins"], N=N, K=K)
    _, _, gx_onehot = make_hue_geometric_tensor(enc_onehot["bins"], N=N, K=K)

    consistency = assert_group_consistency(
        gx_soft,
        r2_act,
        hue_step=1,
        rot_step=1 if N > 1 else 0,
        strict=strict_assertions,
    )

    dec_soft_id = decode_huebins_to_rgb(
        gx_soft.tensor, enc_soft["sat"], enc_soft["val"], K, mode=decoding_mode, n_rotation_bins=N
    )["rgb"]
    dec_onehot_id = decode_huebins_to_rgb(
        gx_onehot.tensor, enc_onehot["sat"], enc_onehot["val"], K, mode=decoding_mode, n_rotation_bins=N
    )["rgb"]

    mae_soft = compute_mae(rgb, dec_soft_id)
    psnr_soft = compute_psnr(rgb, dec_soft_id)
    mae_onehot = compute_mae(rgb, dec_onehot_id)
    psnr_onehot = compute_psnr(rgb, dec_onehot_id)

    images = [rgb.permute(1, 2, 0).cpu().numpy(), dec_soft_id.permute(1, 2, 0).cpu().numpy()]
    titles = ["Synthetic RGB", "Decoded identity (soft)"]

    for step in hue_steps:
        g = build_group_elements(r2_act, hue_step=int(step), rot_step=0)["combined"]
        img = decode_huebins_to_rgb(
            gx_soft.transform(g).tensor,
            enc_soft["sat"],
            enc_soft["val"],
            K,
            mode=decoding_mode,
            n_rotation_bins=N,
        )["rgb"]
        images.append(img.permute(1, 2, 0).cpu().numpy())
        titles.append(f"Hue step {int(step)}")

    if show_plot:
        _plot_image_row(images, titles, figsize=(3 * len(images), 3))

    return {
        "mae_identity_soft": mae_soft,
        "psnr_identity_soft": psnr_soft,
        "mae_identity_onehot": mae_onehot,
        "psnr_identity_onehot": psnr_onehot,
        "mae_delta_onehot_minus_soft": mae_onehot - mae_soft,
        "group_consistency": consistency,
    }


def run_pathmnist_hue_validation(
    dataset,
    sample_idx: int = 0,
    N: int = 1,
    K: int = 50,
    hue_steps: Sequence[int] = (0, 1, 2, 3),
    encoding_mode: str = "soft",
    decoding_mode: str = "circular_mean",
    strict_assertions: bool = True,
    show_plot: bool = True,
) -> Dict:
    sample = dataset[sample_idx]
    if not isinstance(sample, (tuple, list)) or len(sample) == 0:
        raise ValueError("Dataset sample must return (image, label) or similar tuple/list")
    rgb = sample[0].float()

    enc_soft = encode_rgb_to_huebins(rgb, K, mode=encoding_mode)
    enc_onehot = encode_rgb_to_huebins(rgb, K, mode="onehot")

    r2_act, _, gx_soft = make_hue_geometric_tensor(enc_soft["bins"], N=N, K=K)
    _, _, gx_onehot = make_hue_geometric_tensor(enc_onehot["bins"], N=N, K=K)

    consistency = assert_group_consistency(
        gx_soft,
        r2_act,
        hue_step=1,
        rot_step=1 if N > 1 else 0,
        strict=strict_assertions,
    )

    dec_soft_id = decode_huebins_to_rgb(
        gx_soft.tensor, enc_soft["sat"], enc_soft["val"], K, mode=decoding_mode, n_rotation_bins=N
    )["rgb"]
    dec_onehot_id = decode_huebins_to_rgb(
        gx_onehot.tensor, enc_onehot["sat"], enc_onehot["val"], K, mode=decoding_mode, n_rotation_bins=N
    )["rgb"]

    mae_soft = compute_mae(rgb, dec_soft_id)
    psnr_soft = compute_psnr(rgb, dec_soft_id)
    mae_onehot = compute_mae(rgb, dec_onehot_id)
    psnr_onehot = compute_psnr(rgb, dec_onehot_id)

    images = [rgb.permute(1, 2, 0).cpu().numpy(), dec_soft_id.permute(1, 2, 0).cpu().numpy()]
    titles = ["Original RGB", "Decoded identity (soft)"]

    for step in hue_steps:
        g = build_group_elements(r2_act, hue_step=int(step), rot_step=0)["combined"]
        img = decode_huebins_to_rgb(
            gx_soft.transform(g).tensor,
            enc_soft["sat"],
            enc_soft["val"],
            K,
            mode=decoding_mode,
            n_rotation_bins=N,
        )["rgb"]
        images.append(img.permute(1, 2, 0).cpu().numpy())
        titles.append(f"Hue step {int(step)}")

    if show_plot:
        _plot_image_row(images, titles, figsize=(3 * len(images), 3))

    return {
        "mae_identity_soft": mae_soft,
        "psnr_identity_soft": psnr_soft,
        "mae_identity_onehot": mae_onehot,
        "psnr_identity_onehot": psnr_onehot,
        "mae_delta_onehot_minus_soft": mae_onehot - mae_soft,
        "group_consistency": consistency,
    }


def run_rotation_hue_combined_demo(
    rgb: torch.Tensor,
    N: int = 4,
    K: int = 4,
    hue_step: int = 1,
    rot_step: int = 1,
    encoding_mode: str = "soft",
    decoding_mode: str = "circular_mean",
    show_plot: bool = True,
) -> Dict:
    if N <= 1:
        raise ValueError("run_rotation_hue_combined_demo requires N > 1")

    enc = encode_rgb_to_huebins(rgb, K, mode=encoding_mode)
    r2_act, _, gx = make_hue_geometric_tensor(enc["bins"], N=N, K=K)

    elems_h = build_group_elements(r2_act, hue_step=hue_step, rot_step=0)
    elems_r = build_group_elements(r2_act, hue_step=0, rot_step=rot_step)
    elems_b = build_group_elements(r2_act, hue_step=hue_step, rot_step=rot_step)

    rgb_h = decode_huebins_to_rgb(
        gx.transform(elems_h["combined"]).tensor,
        enc["sat"],
        enc["val"],
        K,
        mode=decoding_mode,
        n_rotation_bins=N,
    )["rgb"]
    rgb_r = decode_huebins_to_rgb(
        gx.transform(elems_r["combined"]).tensor,
        enc["sat"],
        enc["val"],
        K,
        mode=decoding_mode,
        n_rotation_bins=N,
    )["rgb"]
    rgb_b = decode_huebins_to_rgb(
        gx.transform(elems_b["combined"]).tensor,
        enc["sat"],
        enc["val"],
        K,
        mode=decoding_mode,
        n_rotation_bins=N,
    )["rgb"]

    if show_plot:
        images = [
            rgb.permute(1, 2, 0).cpu().numpy(),
            rgb_h.permute(1, 2, 0).cpu().numpy(),
            rgb_r.permute(1, 2, 0).cpu().numpy(),
            rgb_b.permute(1, 2, 0).cpu().numpy(),
        ]
        titles = [
            "Original RGB",
            f"Hue only (+{hue_step})",
            f"Rotation only (+{rot_step})",
            "Rotation + Hue",
        ]
        _plot_image_row(images, titles, figsize=(12, 3))

    return {
        "note": "Rotation plots are expected to show interpolation artifacts on pixel grids.",
        "hue_step": int(hue_step),
        "rot_step": int(rot_step),
        "rot_order": int(N),
        "hue_order": int(K),
    }
