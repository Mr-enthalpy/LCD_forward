from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(dirpath: Path) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    return dirpath


def normalize_for_display(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-10:
        return np.zeros_like(img)
    return (img - vmin) / (vmax - vmin)


def plot_mask_grid(
    masks: np.ndarray,
    mask_ids: list,
    out_path: Path,
    ncols: int = 6,
    title: str = "选定掩膜",
) -> Path:
    n = masks.shape[0]
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.atleast_2d(axes)

    for idx in range(n):
        r, c = divmod(idx, ncols)
        mask_2d = masks[idx].squeeze()
        axes[r, c].imshow(mask_2d, cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_title(f"掩膜 {idx + 1}", fontsize=7)
        axes[r, c].axis("off")

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_psf_panel(
    psfs: np.ndarray,
    wavelengths_nm: np.ndarray,
    out_path: Path,
    title: str = "PSF 面板",
    psfs_pred: np.ndarray | None = None,
) -> Path:
    n_examples = psfs.shape[0]
    n_wl = len(wavelengths_nm)
    n_cols = n_wl if psfs_pred is None else 2 * n_wl + 1
    n_rows = n_examples

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = np.atleast_2d(axes)

    for i in range(n_examples):
        for j in range(n_wl):
            psf_img = psfs[i, 0, j] if psfs.ndim >= 5 else psfs[i, j]
            axes[i, j].imshow(normalize_for_display(psf_img), cmap="hot")
            if i == 0:
                axes[i, j].set_title(f"{wavelengths_nm[j]:.0f} nm 实测", fontsize=8)
            axes[i, j].axis("off")

        if psfs_pred is not None:
            axes[i, n_wl].axis("off")
            for j in range(n_wl):
                col = n_wl + 1 + j
                pred_img = psfs_pred[i, 0, j] if psfs_pred.ndim >= 5 else psfs_pred[i, j]
                axes[i, col].imshow(normalize_for_display(pred_img), cmap="hot")
                if i == 0:
                    axes[i, col].set_title(f"{wavelengths_nm[j]:.0f} nm 预测", fontsize=8)
                axes[i, col].axis("off")

            for j in range(n_wl):
                col = n_wl + 1 + j
                gt_img = psfs[i, 0, j] if psfs.ndim >= 5 else psfs[i, j]
                pred_img = psfs_pred[i, 0, j] if psfs_pred.ndim >= 5 else psfs_pred[i, j]
                err = np.abs(gt_img - pred_img)
                axes[i, col].imshow(normalize_for_display(err), cmap="inferno")
                if i == 0:
                    axes[i, col].set_title(f"{wavelengths_nm[j]:.0f} nm 误差", fontsize=8)
                axes[i, col].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_measured_vs_predicted(
    gt_psfs: np.ndarray,
    pred_psfs: np.ndarray,
    mask_ids: list,
    wavelengths_nm: np.ndarray,
    out_path: Path,
    title: str = "实测与预测 PSF 对比",
) -> Path:
    n_examples = gt_psfs.shape[0]
    n_wl = len(wavelengths_nm)

    fig, axes = plt.subplots(n_examples, 3 * n_wl, figsize=(3 * n_wl * 2.2, n_examples * 2.2))
    if n_examples == 1:
        axes = np.atleast_2d(axes)

    for i in range(n_examples):
        gt_img = gt_psfs[i, 0] if gt_psfs.ndim >= 5 else gt_psfs[i]
        pred_img = pred_psfs[i, 0] if pred_psfs.ndim >= 5 else pred_psfs[i]

        for j in range(n_wl):
            axes[i, j].imshow(normalize_for_display(gt_img[j]), cmap="hot")
            if i == 0:
                axes[i, j].set_title(f"实测 {wavelengths_nm[j]:.0f}nm", fontsize=7)
            axes[i, j].axis("off")

        for j in range(n_wl):
            col = n_wl + j
            axes[i, col].imshow(normalize_for_display(pred_img[j]), cmap="hot")
            if i == 0:
                axes[i, col].set_title(f"预测 {wavelengths_nm[j]:.0f}nm", fontsize=7)
            axes[i, col].axis("off")

        for j in range(n_wl):
            col = 2 * n_wl + j
            err = np.abs(gt_img[j] - pred_img[j])
            axes[i, col].imshow(normalize_for_display(err), cmap="inferno")
            if i == 0:
                axes[i, col].set_title(f"误差 {wavelengths_nm[j]:.0f}nm", fontsize=7)
            axes[i, col].axis("off")

        axes[i, 0].set_ylabel(f"样本 {i + 1}", fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_pca_basis_preview(
    components: np.ndarray,
    explained_variance: np.ndarray,
    out_path: Path,
    n_preview: int = 12,
) -> Path:
    n_comp = min(n_preview, components.shape[0])
    ncols = 6
    nrows = max(1, (n_comp + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.atleast_2d(axes)

    for i in range(n_comp):
        r, c = divmod(i, ncols)
        comp = components[i].reshape(int(np.sqrt(components.shape[1])), -1)
        axes[r, c].imshow(comp, cmap="RdBu_r", vmin=-np.abs(comp).max(), vmax=np.abs(comp).max())
        axes[r, c].set_title(f"主成分 {i + 1} ({explained_variance[i]:.3f})", fontsize=7)
        axes[r, c].axis("off")

    for i in range(n_comp, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    fig.suptitle("PSF 的 PCA 基底预览", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_synthetic_objects(objects: np.ndarray, out_path: Path) -> Path:
    n_ch = objects.shape[0]
    plt.figure(figsize=(n_ch * 3, 3))
    for c in range(n_ch):
        plt.subplot(1, n_ch, c + 1)
        plt.imshow(objects[c], cmap="viridis")
        plt.title(f"通道 {c + 1}", fontsize=10)
        plt.axis("off")
    plt.suptitle("合成目标物体", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_rendered_frames(frames: np.ndarray, out_path: Path, title: str = "渲染观测帧") -> Path:
    n_frames = frames.shape[0]
    ncols = min(6, n_frames)
    nrows = max(1, (n_frames + ncols - 1) // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = np.atleast_2d(axes)

    for t in range(n_frames):
        r, c = divmod(t, ncols)
        axes[r, c].imshow(frames[t], cmap="gray")
        axes[r, c].set_title(f"帧 {t + 1}", fontsize=8)
        axes[r, c].axis("off")

    for t in range(n_frames, nrows * ncols):
        r, c = divmod(t, ncols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reconstruction(
    gt: np.ndarray,
    recon: np.ndarray,
    out_path: Path,
    wavelengths_nm: np.ndarray | None = None,
    title: str = "重建结果",
) -> Path:
    n_ch = gt.shape[0]
    plt.figure(figsize=(n_ch * 4, 8))

    for c in range(n_ch):
        plt.subplot(3, n_ch, c + 1)
        plt.imshow(gt[c], cmap="viridis")
        wl_label = f" ({wavelengths_nm[c]:.0f}nm)" if wavelengths_nm is not None else ""
        plt.title(f"真实 通道{c + 1}{wl_label}", fontsize=9)
        plt.axis("off")

        plt.subplot(3, n_ch, n_ch + c + 1)
        plt.imshow(recon[c], cmap="viridis")
        plt.title(f"重建 通道{c + 1}{wl_label}", fontsize=9)
        plt.axis("off")

        err = np.abs(gt[c] - recon[c])
        plt.subplot(3, n_ch, 2 * n_ch + c + 1)
        plt.imshow(err, cmap="inferno")
        plt.title(f"误差 通道{c + 1}", fontsize=9)
        plt.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_recon_comparison(
    gt: np.ndarray,
    recon_single: np.ndarray,
    recon_multi: np.ndarray,
    out_path: Path,
    wavelengths_nm: np.ndarray | None = None,
) -> Path:
    n_ch = gt.shape[0]
    fig, axes = plt.subplots(4, n_ch, figsize=(n_ch * 4.2, 12.8), squeeze=False)
    row_labels = [
        "真实目标",
        "单帧\n重建",
        "多帧\n重建",
        "绝对误差\n（单帧 | 多帧）",
    ]

    for c in range(n_ch):
        channel_label = f"{wavelengths_nm[c]:.0f} nm" if wavelengths_nm is not None else f"通道 {c + 1}"

        axes[0, c].imshow(gt[c], cmap="viridis")
        axes[0, c].set_title(channel_label, fontsize=15, fontweight="bold")
        axes[0, c].axis("off")

        axes[1, c].imshow(recon_single[c], cmap="viridis")
        axes[1, c].axis("off")

        axes[2, c].imshow(recon_multi[c], cmap="viridis")
        axes[2, c].axis("off")

        err_s = np.abs(gt[c] - recon_single[c])
        err_m = np.abs(gt[c] - recon_multi[c])
        combined_err = np.hstack([err_s, err_m])
        axes[3, c].imshow(combined_err, cmap="inferno")
        axes[3, c].set_title("单帧 | 多帧", fontsize=14)
        axes[3, c].axvline(x=err_s.shape[1] - 0.5, color="white", linewidth=1.2)
        axes[3, c].axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.12,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            fontsize=14,
            rotation=0,
            va="center",
            ha="right",
            clip_on=False,
        )

    fig.suptitle("逐波段重建对比", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0.02, 1.0, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _rgb_indices(wavelengths_nm: np.ndarray | None, n_channels: int) -> list[int]:
    if wavelengths_nm is None:
        return [min(n_channels - 1, i) for i in (2, 1, 0)]
    wavelengths = np.asarray(wavelengths_nm, dtype=np.float32)
    targets = np.array([650.0, 550.0, 450.0], dtype=np.float32)
    return [int(np.argmin(np.abs(wavelengths - target))) for target in targets]


def _to_pseudorgb(cube: np.ndarray, wavelengths_nm: np.ndarray | None = None) -> np.ndarray:
    if cube.ndim != 3:
        raise ValueError(f"Expected spectral cube [L,H,W], got shape {cube.shape}")
    indices = _rgb_indices(wavelengths_nm, cube.shape[0])
    rgb = np.stack([cube[i] for i in indices], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [1.0, 99.0])
    scale = hi - lo if hi > lo else 1.0
    return np.clip((rgb - lo) / scale, 0.0, 1.0)


def plot_recon_rgb_pseudocolor_comparison(
    gt: np.ndarray,
    recon_single: np.ndarray,
    recon_multi: np.ndarray,
    out_path: Path,
    wavelengths_nm: np.ndarray | None = None,
) -> Path:
    gt_rgb = _to_pseudorgb(gt, wavelengths_nm)
    single_rgb = _to_pseudorgb(recon_single, wavelengths_nm)
    multi_rgb = _to_pseudorgb(recon_multi, wavelengths_nm)
    err_single = np.mean(np.abs(gt_rgb - single_rgb), axis=-1)
    err_multi = np.mean(np.abs(gt_rgb - multi_rgb), axis=-1)
    err_vmax = float(np.percentile(np.concatenate([err_single.ravel(), err_multi.ravel()]), 99.0))
    if err_vmax <= 0.0:
        err_vmax = 1.0

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.8), squeeze=False)
    panels = [
        (0, 0, gt_rgb, "真实目标", None, None),
        (0, 1, single_rgb, "单帧重建", None, None),
        (0, 2, multi_rgb, "多帧重建", None, None),
        (1, 0, err_single, "绝对误差：单帧", "inferno", err_vmax),
        (1, 1, err_multi, "绝对误差：多帧", "inferno", err_vmax),
    ]
    for row_idx, col_idx, image, title, cmap, vmax in panels:
        if vmax is None:
            axes[row_idx, col_idx].imshow(image, cmap=cmap)
        else:
            axes[row_idx, col_idx].imshow(image, cmap=cmap, vmin=0.0, vmax=vmax)
        axes[row_idx, col_idx].set_title(title, fontsize=15, fontweight="bold")
        axes[row_idx, col_idx].axis("off")

    axes[1, 2].axis("off")
    axes[0, 0].text(
        -0.12,
        0.5,
        "伪彩色",
        transform=axes[0, 0].transAxes,
        fontsize=14,
        rotation=0,
        va="center",
        ha="right",
        clip_on=False,
    )
    axes[1, 0].text(
        -0.12,
        0.5,
        "绝对\n误差",
        transform=axes[1, 0].transAxes,
        fontsize=14,
        rotation=0,
        va="center",
        ha="right",
        clip_on=False,
    )

    fig.suptitle("伪彩色重建对比", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0.06, 0.02, 1.0, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
