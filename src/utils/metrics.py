from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def psnr_from_mse(mse, data_range=1.0):
    return float(10.0 * math.log10((data_range ** 2) / max(float(mse), 1e-12)))


def psnr(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    mse = float(np.mean((gt - pred) ** 2))
    return psnr_from_mse(mse), mse


def correlation(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_vec = gt.ravel().astype(np.float64)
    pred_vec = pred.ravel().astype(np.float64)
    gt_vec = gt_vec - gt_vec.mean()
    pred_vec = pred_vec - pred_vec.mean()
    denom = np.sqrt(np.sum(gt_vec ** 2) * np.sum(pred_vec ** 2)) + 1e-12
    return float(np.sum(gt_vec * pred_vec) / denom)


def box_mean(image: np.ndarray, window: int = 11) -> np.ndarray:
    pad = window // 2
    padded = np.pad(image.astype(np.float64), pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return windows.mean(axis=(-1, -2))


def ssim_box11(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_gt = box_mean(gt)
    mu_pred = box_mean(pred)
    mu_gt2 = mu_gt * mu_gt
    mu_pred2 = mu_pred * mu_pred
    mu_cross = mu_gt * mu_pred
    sigma_gt = box_mean(gt.astype(np.float64) * gt.astype(np.float64)) - mu_gt2
    sigma_pred = box_mean(pred.astype(np.float64) * pred.astype(np.float64)) - mu_pred2
    sigma_cross = box_mean(gt.astype(np.float64) * pred.astype(np.float64)) - mu_cross
    score = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
        (mu_gt2 + mu_pred2 + c1) * (sigma_gt + sigma_pred + c2) + 1e-12
    )
    return float(np.mean(score))


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (x_max - x_min)


def compute_channel_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    raw_psnr_val, raw_mse = psnr(gt, pred)
    corr = correlation(gt, pred)
    raw_ssim = ssim_box11(gt, pred)

    gt_display = minmax_normalize(gt)
    pred_display = minmax_normalize(pred)
    display_ssim = ssim_box11(gt_display, pred_display)

    return {
        "mse": raw_mse,
        "psnr": raw_psnr_val,
        "correlation": corr,
        "ssim_raw": raw_ssim,
        "ssim_display": display_ssim,
    }
