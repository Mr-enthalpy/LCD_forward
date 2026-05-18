from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _psf_energy(x: torch.Tensor) -> torch.Tensor:
    return x.sum(dim=(-2, -1))


def _psf_centroid(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., H, W]
    return: [..., 2] where last dim is (cy, cx)
    """
    h, w = x.shape[-2:]
    device = x.device
    dtype = x.dtype

    yy = torch.arange(h, device=device, dtype=dtype).view(1, 1, h, 1)
    xx = torch.arange(w, device=device, dtype=dtype).view(1, 1, 1, w)

    mass = x.sum(dim=(-2, -1), keepdim=True) + 1e-8
    cy = (x * yy).sum(dim=(-2, -1), keepdim=True) / mass
    cx = (x * xx).sum(dim=(-2, -1), keepdim=True) / mass

    return torch.cat([cy, cx], dim=-1).squeeze(-2)  # [..., 2]


def forward_loss(
    pred_psfs: torch.Tensor,
    gt_psfs: torch.Tensor,
    w_l1: float = 1.0,
    w_mse: float = 0.5,
    w_fft: float = 0.1,
    w_energy: float = 0.0,
    w_centroid: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Forward-model loss for PSF prediction.

    Args:
        pred_psfs: [B, T, L, H, W] or compatible
        gt_psfs:   [B, T, L, H, W] or compatible

    Returns:
        dict with:
            "loss"
            "loss_l1"
            "loss_mse"
            "loss_fft"
            "loss_energy"
            "loss_centroid"
    """
    if pred_psfs.shape != gt_psfs.shape:
        raise ValueError(
            f"Shape mismatch in forward_loss: pred={pred_psfs.shape}, gt={gt_psfs.shape}"
        )

    loss_l1 = F.l1_loss(pred_psfs, gt_psfs)
    loss_mse = F.mse_loss(pred_psfs, gt_psfs)

    fft_pred = torch.fft.fft2(pred_psfs, dim=(-2, -1))
    fft_gt = torch.fft.fft2(gt_psfs, dim=(-2, -1))
    loss_fft = F.l1_loss(torch.abs(fft_pred), torch.abs(fft_gt))

    energy_pred = _psf_energy(pred_psfs)
    energy_gt = _psf_energy(gt_psfs)
    loss_energy = F.l1_loss(energy_pred, energy_gt)

    centroid_pred = _psf_centroid(pred_psfs)
    centroid_gt = _psf_centroid(gt_psfs)
    loss_centroid = F.l1_loss(centroid_pred, centroid_gt)

    loss = (
        w_l1 * loss_l1
        + w_mse * loss_mse
        + w_fft * loss_fft
        + w_energy * loss_energy
        + w_centroid * loss_centroid
    )

    return {
        "loss": loss,
        "loss_l1": loss_l1.detach(),
        "loss_mse": loss_mse.detach(),
        "loss_fft": loss_fft.detach(),
        "loss_energy": loss_energy.detach(),
        "loss_centroid": loss_centroid.detach(),
    }


def recon_loss(
    pred_objects: torch.Tensor,
    gt_objects: torch.Tensor,
    w_l1: float = 1.0,
    w_mse: float = 0.5,
    w_tv: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Reconstruction loss.

    Args:
        pred_objects: [B, L, H, W]
        gt_objects:   [B, L, H, W]
    """
    if pred_objects.shape != gt_objects.shape:
        raise ValueError(
            f"Shape mismatch in recon_loss: pred={pred_objects.shape}, gt={gt_objects.shape}"
        )

    loss_l1 = F.l1_loss(pred_objects, gt_objects)
    loss_mse = F.mse_loss(pred_objects, gt_objects)

    if w_tv > 0.0:
        tv_h = torch.abs(pred_objects[..., 1:, :] - pred_objects[..., :-1, :]).mean()
        tv_w = torch.abs(pred_objects[..., :, 1:] - pred_objects[..., :, :-1]).mean()
        loss_tv = tv_h + tv_w
    else:
        loss_tv = pred_objects.new_tensor(0.0)

    loss = w_l1 * loss_l1 + w_mse * loss_mse + w_tv * loss_tv

    return {
        "loss": loss,
        "loss_l1": loss_l1.detach(),
        "loss_mse": loss_mse.detach(),
        "loss_tv": loss_tv.detach(),
    }


def psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return 10.0 * torch.log10((data_range ** 2) / (mse + 1e-12))


@torch.no_grad()
def recon_metrics(
    pred_objects: torch.Tensor,
    gt_objects: torch.Tensor,
    data_range: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Lightweight reconstruction metrics for eval logging.
    """
    if pred_objects.shape != gt_objects.shape:
        raise ValueError(
            f"Shape mismatch in recon_metrics: pred={pred_objects.shape}, gt={gt_objects.shape}"
        )

    mae = F.l1_loss(pred_objects, gt_objects)
    mse = F.mse_loss(pred_objects, gt_objects)
    psnr = psnr_from_mse(mse, data_range=data_range)

    return {
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
    }