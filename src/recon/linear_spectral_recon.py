from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.utils.metrics import compute_channel_metrics


def resize_psf_tensor(psf: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
    if psf.shape[-2:] == target_size:
        return psf
    psf = psf.unsqueeze(0).unsqueeze(0) if psf.ndim == 2 else psf
    psf = torch.nn.functional.interpolate(
        psf, size=target_size, mode="bilinear", align_corners=False
    )
    psf = psf / (psf.sum(dim=(-2, -1), keepdim=True) + 1e-8)
    return psf


def generate_synthetic_object(
    n_channels: int = 3,
    spatial_size: Tuple[int, int] = (256, 256),
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = spatial_size
    obj = np.zeros((n_channels, h, w), dtype=np.float32)

    for c in range(n_channels):
        if c == 0:
            num_bars = rng.integers(3, 8)
            for _ in range(num_bars):
                x0 = rng.integers(0, w // 2)
                width = rng.integers(4, 16)
                intensity = rng.uniform(0.4, 1.0)
                obj[c, :, x0:x0 + width] = intensity
            yy, xx = np.meshgrid(
                np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij"
            )
            obj[c] += 0.15 * (1.0 + np.sin(2 * np.pi * 8 * xx))

        elif c == 1:
            for _ in range(rng.integers(3, 6)):
                cy = rng.integers(h // 4, 3 * h // 4)
                cx = rng.integers(w // 4, 3 * w // 4)
                r = rng.integers(h // 16, h // 6)
                intensity = rng.uniform(0.5, 1.0)
                yy, xx = np.ogrid[:h, :w]
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                obj[c][mask] = intensity

        else:
            for _ in range(rng.integers(5, 10)):
                y0 = rng.integers(0, 3 * h // 4)
                x0 = rng.integers(0, 3 * w // 4)
                bh = rng.integers(h // 16, h // 8)
                bw = rng.integers(w // 16, w // 8)
                intensity = rng.uniform(0.4, 1.0)
                obj[c, y0:y0 + bh, x0:x0 + bw] = intensity
            obj[c] += 0.1 * rng.random((h, w), dtype=np.float32)

    obj = obj / (obj.max() + 1e-8)
    return obj


def render_frames_fft(
    objects: torch.Tensor,
    psfs: torch.Tensor,
) -> torch.Tensor:
    if objects.ndim != 3:
        raise ValueError(f"objects must be [L, H, W], got {objects.shape}")
    if psfs.ndim != 4:
        raise ValueError(f"psfs must be [T, L, Hp, Wp], got {psfs.shape}")

    n_lambda, h, w = objects.shape
    n_frames, n_lambda2, hp, wp = psfs.shape

    if n_lambda != n_lambda2:
        raise ValueError(f"Lambda mismatch: objects L={n_lambda}, psfs L={n_lambda2}")

    if (h, w) != (hp, wp):
        psfs_list = []
        for t in range(n_frames):
            psfs_l = []
            for l in range(n_lambda):
                psf_resized = resize_psf_tensor(psfs[t, l], (h, w))
                psfs_l.append(psf_resized)
            psfs_list.append(torch.stack(psfs_l, dim=0))
        psfs = torch.stack(psfs_list, dim=0)

    frames = torch.zeros(n_frames, h, w, dtype=objects.dtype, device=objects.device)
    obj_fft = torch.fft.fft2(objects)

    for t in range(n_frames):
        acc = torch.zeros(h, w, dtype=torch.complex64, device=objects.device)
        for l in range(n_lambda):
            psf_fft = torch.fft.fft2(psfs[t, l])
            acc = acc + psf_fft * obj_fft[l]
        frames[t] = torch.fft.ifft2(acc).real

    return frames


def _resize_psfs_for_recon(psfs, target_h, target_w, n_lambda, n_frames, device):
    hp, wp = psfs.shape[-2:]
    if (target_h, target_w) != (hp, wp):
        psfs_list = []
        for t in range(n_frames):
            psfs_l = []
            for l in range(n_lambda):
                psf_resized = resize_psf_tensor(psfs[t, l], (target_h, target_w))
                psfs_l.append(psf_resized)
            psfs_list.append(torch.stack(psfs_l, dim=0))
        return torch.stack(psfs_list, dim=0)
    return psfs


def _solve_per_frequency(H, y, alpha, policy, eye, cond_threshold=1e4):
    HTH = H.conj().T @ H
    Hty = H.conj().T @ y
    H_norm = (HTH.abs().max() + 1e-12).real

    if policy == "none" or alpha == 0.0:
        try:
            return torch.linalg.solve(HTH, Hty)
        except RuntimeError:
            s = torch.linalg.svdvals(HTH)
            rcond = (s[-1] / (s[0] + 1e-12)).real
            min_reg = 1e-12 * H_norm
            return torch.linalg.solve(HTH + min_reg * eye, Hty)

    elif policy == "threshold":
        s = torch.linalg.svdvals(HTH)
        cond = (s[0] / (s[-1] + 1e-12)).real
        if cond < cond_threshold:
            return torch.linalg.solve(HTH, Hty)
        else:
            reg = alpha * H_norm * eye
            return torch.linalg.solve(HTH + reg, Hty)

    elif policy == "adaptive":
        reg = alpha * H_norm * eye
        return torch.linalg.solve(HTH + reg, Hty)

    else:
        reg = alpha * H_norm * eye
        return torch.linalg.solve(HTH + reg, Hty)


def frequency_domain_ridge_reconstruct(
    frames: torch.Tensor,
    psfs: torch.Tensor,
    alpha: float = 1e-4,
    policy: str = "adaptive",
    cond_threshold: float = 1e3,
) -> torch.Tensor:
    if frames.ndim != 3:
        raise ValueError(f"frames must be [T, H, W], got {frames.shape}")
    if psfs.ndim != 4:
        raise ValueError(f"psfs must be [T, L, Hp, Wp], got {psfs.shape}")

    n_frames, h, w = frames.shape
    n_frames2, n_lambda, hp, wp = psfs.shape

    if n_frames != n_frames2:
        raise ValueError(f"Frame mismatch: frames T={n_frames}, psfs T={n_frames2}")

    psfs = _resize_psfs_for_recon(psfs, h, w, n_lambda, n_frames, frames.device)

    reconstruction = torch.zeros(n_lambda, h, w, dtype=torch.complex128, device=frames.device)
    frames_fft = torch.fft.fft2(frames.double())
    psfs_fft = torch.fft.fft2(psfs.double())

    eye = torch.eye(n_lambda, dtype=torch.complex128, device=frames.device)

    for i in range(h):
        for j in range(w):
            H = psfs_fft[:, :, i, j]
            y = frames_fft[:, i, j]
            reconstruction[:, i, j] = _solve_per_frequency(H, y, alpha, policy, eye, cond_threshold)

    for c in range(n_lambda):
        reconstruction[c] = torch.fft.ifft2(reconstruction[c]).real

    reconstruction = torch.clamp(reconstruction.real, 0.0, None)
    return reconstruction.float()


def single_frame_reconstruct(
    frames: torch.Tensor,
    psfs: torch.Tensor,
    alpha: float = 1e-4,
    policy: str = "adaptive",
    cond_threshold: float = 1e3,
) -> torch.Tensor:
    if frames.ndim == 3 and frames.shape[0] > 1:
        frames = frames[0:1]
    if psfs.ndim == 4 and psfs.shape[0] > 1:
        psfs = psfs[0:1]
    return frequency_domain_ridge_reconstruct(frames, psfs, alpha=alpha, policy=policy, cond_threshold=cond_threshold)


def compute_recon_metrics(
    gt_object: torch.Tensor,
    pred_object: torch.Tensor,
) -> Dict:
    gt = gt_object.detach().float()
    pred = pred_object.detach().float()

    n_channels = gt.shape[0]

    metrics = {"per_channel": {}, "mean": {}}

    for c in range(n_channels):
        gt_c = gt[c].ravel()
        pred_c = pred[c].ravel()

        mse = float(torch.mean((gt_c - pred_c) ** 2))
        rel_l2 = float(
            torch.sqrt(torch.sum((gt_c - pred_c) ** 2))
            / (torch.sqrt(torch.sum(gt_c ** 2)) + 1e-8)
        )
        import math
        psnr_val = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))

        gt_centered = gt_c - gt_c.mean()
        pred_centered = pred_c - pred_c.mean()
        denom = torch.sqrt(torch.sum(gt_centered ** 2) * torch.sum(pred_centered ** 2)) + 1e-8
        corr = float(torch.sum(gt_centered * pred_centered) / denom)

        gt_ch_np = gt[c].cpu().numpy()
        pred_ch_np = pred[c].cpu().numpy()
        ch_metrics = compute_channel_metrics(gt_ch_np, pred_ch_np)

        metrics["per_channel"][str(c)] = {
            "mse": mse,
            "relative_l2": rel_l2,
            "psnr": psnr_val,
            "correlation": corr,
            "ssim_raw": ch_metrics["ssim_raw"],
            "ssim_display": ch_metrics["ssim_display"],
        }

    all_mse = [metrics["per_channel"][str(c)]["mse"] for c in range(n_channels)]
    all_rel = [metrics["per_channel"][str(c)]["relative_l2"] for c in range(n_channels)]
    all_psnr = [metrics["per_channel"][str(c)]["psnr"] for c in range(n_channels)]
    all_corr = [metrics["per_channel"][str(c)]["correlation"] for c in range(n_channels)]
    all_ssim_raw = [metrics["per_channel"][str(c)]["ssim_raw"] for c in range(n_channels)]
    all_ssim_display = [metrics["per_channel"][str(c)]["ssim_display"] for c in range(n_channels)]

    metrics["mean"] = {
        "mse": float(np.mean(all_mse)),
        "relative_l2": float(np.mean(all_rel)),
        "psnr": float(np.mean(all_psnr)),
        "correlation": float(np.mean(all_corr)),
        "ssim_raw": float(np.mean(all_ssim_raw)),
        "ssim_display": float(np.mean(all_ssim_display)),
    }

    return metrics
