from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


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
    title: str = "Selected Masks",
) -> Path:
    n = masks.shape[0]
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = np.atleast_2d(axes)

    for idx in range(n):
        r, c = divmod(idx, ncols)
        mask_2d = masks[idx].squeeze()
        axes[r, c].imshow(mask_2d, cmap="gray", vmin=0, vmax=1)
        axes[r, c].set_title(mask_ids[idx], fontsize=7)
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
    title: str = "PSF Panel",
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
                axes[i, j].set_title(f"{wavelengths_nm[j]:.0f} nm GT", fontsize=8)
            axes[i, j].axis("off")

        if psfs_pred is not None:
            axes[i, n_wl].axis("off")
            for j in range(n_wl):
                col = n_wl + 1 + j
                pred_img = psfs_pred[i, 0, j] if psfs_pred.ndim >= 5 else psfs_pred[i, j]
                axes[i, col].imshow(normalize_for_display(pred_img), cmap="hot")
                if i == 0:
                    axes[i, col].set_title(f"{wavelengths_nm[j]:.0f} nm Pred", fontsize=8)
                axes[i, col].axis("off")

            for j in range(n_wl):
                col = n_wl + 1 + j
                gt_img = psfs[i, 0, j] if psfs.ndim >= 5 else psfs[i, j]
                pred_img = psfs_pred[i, 0, j] if psfs_pred.ndim >= 5 else psfs_pred[i, j]
                err = np.abs(gt_img - pred_img)
                axes[i, col].imshow(normalize_for_display(err), cmap="inferno")
                if i == 0:
                    axes[i, col].set_title(f"{wavelengths_nm[j]:.0f} nm Err", fontsize=8)
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
    title: str = "Measured vs Predicted PSFs",
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
                axes[i, j].set_title(f"GT {wavelengths_nm[j]:.0f}nm", fontsize=7)
            axes[i, j].axis("off")

        for j in range(n_wl):
            col = n_wl + j
            axes[i, col].imshow(normalize_for_display(pred_img[j]), cmap="hot")
            if i == 0:
                axes[i, col].set_title(f"Pred {wavelengths_nm[j]:.0f}nm", fontsize=7)
            axes[i, col].axis("off")

        for j in range(n_wl):
            col = 2 * n_wl + j
            err = np.abs(gt_img[j] - pred_img[j])
            axes[i, col].imshow(normalize_for_display(err), cmap="inferno")
            if i == 0:
                axes[i, col].set_title(f"Err {wavelengths_nm[j]:.0f}nm", fontsize=7)
            axes[i, col].axis("off")

        axes[i, 0].set_ylabel(mask_ids[i], fontsize=7)

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
        axes[r, c].set_title(f"PC{i + 1} ({explained_variance[i]:.3f})", fontsize=7)
        axes[r, c].axis("off")

    for i in range(n_comp, nrows * ncols):
        r, c = divmod(i, ncols)
        axes[r, c].axis("off")

    fig.suptitle("PSF PCA Basis Preview", fontsize=12)
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
        plt.title(f"Channel {c}", fontsize=10)
        plt.axis("off")
    plt.suptitle("Synthetic Target Objects", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def plot_rendered_frames(frames: np.ndarray, out_path: Path, title: str = "Rendered Frames") -> Path:
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
        axes[r, c].set_title(f"Frame {t + 1}", fontsize=8)
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
    title: str = "Reconstruction",
) -> Path:
    n_ch = gt.shape[0]
    plt.figure(figsize=(n_ch * 4, 8))

    for c in range(n_ch):
        plt.subplot(3, n_ch, c + 1)
        plt.imshow(gt[c], cmap="viridis")
        wl_label = f" ({wavelengths_nm[c]:.0f}nm)" if wavelengths_nm is not None else ""
        plt.title(f"GT Ch{c}{wl_label}", fontsize=9)
        plt.axis("off")

        plt.subplot(3, n_ch, n_ch + c + 1)
        plt.imshow(recon[c], cmap="viridis")
        plt.title(f"Recon Ch{c}{wl_label}", fontsize=9)
        plt.axis("off")

        err = np.abs(gt[c] - recon[c])
        plt.subplot(3, n_ch, 2 * n_ch + c + 1)
        plt.imshow(err, cmap="inferno")
        plt.title(f"Err Ch{c}", fontsize=9)
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
    plt.figure(figsize=(n_ch * 4, 12))

    for c in range(n_ch):
        wl_label = f" ({wavelengths_nm[c]:.0f}nm)" if wavelengths_nm is not None else ""

        plt.subplot(4, n_ch, c + 1)
        plt.imshow(gt[c], cmap="viridis")
        plt.title(f"GT Ch{c}{wl_label}", fontsize=9)
        plt.axis("off")

        plt.subplot(4, n_ch, n_ch + c + 1)
        plt.imshow(recon_single[c], cmap="viridis")
        plt.title(f"Single Ch{c}{wl_label}", fontsize=9)
        plt.axis("off")

        plt.subplot(4, n_ch, 2 * n_ch + c + 1)
        plt.imshow(recon_multi[c], cmap="viridis")
        plt.title(f"Multi Ch{c}{wl_label}", fontsize=9)
        plt.axis("off")

        err_s = np.abs(gt[c] - recon_single[c])
        err_m = np.abs(gt[c] - recon_multi[c])
        combined_err = np.hstack([err_s, err_m])
        plt.subplot(4, n_ch, 3 * n_ch + c + 1)
        plt.imshow(combined_err, cmap="inferno")
        plt.title(f"Err S | M Ch{c}", fontsize=9)
        plt.axvline(x=err_s.shape[1] - 0.5, color="white", linewidth=1)
        plt.axis("off")

    plt.suptitle("Single-Frame vs Multi-Frame Reconstruction", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path
