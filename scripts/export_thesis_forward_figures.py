from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.optic_handoff import load_psf_dictionary


DEFAULT_DATASET = Path("D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/train.h5")
DEFAULT_FORWARD_ARTIFACTS = Path("outputs/bishe_first_pass/20260521_224543/forward_validation")
DEFAULT_OUT_DIR = Path("outputs/thesis_figures")
WAVELENGTH_RGB_ORDER = [650.0, 550.0, 450.0]
FORWARD_SAMPLE_TEXT_FONTSIZE = 13.8
FORWARD_COLUMN_TITLE_FONTSIZE = 12.0


class NumpyPCARidge:
    def __init__(self, n_components: int = 24, alpha: float = 1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.n_wavelengths = 0
        self.components: list[np.ndarray] = []
        self.explained_variance_ratio: list[np.ndarray] = []
        self.psf_means: list[np.ndarray] = []
        self.coeff_means: list[np.ndarray] = []
        self.ridge_weights: list[np.ndarray] = []
        self.mask_mean: np.ndarray | None = None
        self.mask_std: np.ndarray | None = None
        self.psf_shape: tuple[int, int] | None = None

    def _scale_masks_fit(self, masks: np.ndarray) -> np.ndarray:
        masks_flat = masks.reshape(masks.shape[0], -1).astype(np.float64)
        self.mask_mean = masks_flat.mean(axis=0)
        self.mask_std = masks_flat.std(axis=0)
        self.mask_std[self.mask_std < 1e-12] = 1.0
        return (masks_flat - self.mask_mean) / self.mask_std

    def _scale_masks_transform(self, masks: np.ndarray) -> np.ndarray:
        if self.mask_mean is None or self.mask_std is None:
            raise RuntimeError("Model is not fitted")
        masks_flat = masks.reshape(masks.shape[0], -1).astype(np.float64)
        return (masks_flat - self.mask_mean) / self.mask_std

    def fit(self, train_masks: np.ndarray, train_psfs: np.ndarray) -> dict[str, Any]:
        x_scaled = self._scale_masks_fit(train_masks)
        self.n_wavelengths = train_psfs.shape[2]
        self.psf_shape = tuple(train_psfs.shape[-2:])
        fit_history: dict[str, Any] = {"n_components": self.n_components, "alpha": self.alpha}

        gram = x_scaled @ x_scaled.T
        dual_matrix = gram + self.alpha * np.eye(gram.shape[0], dtype=np.float64)

        for wl_idx in range(self.n_wavelengths):
            psf_flat = train_psfs[:, 0, wl_idx].reshape(train_psfs.shape[0], -1).astype(np.float64)
            psf_mean = psf_flat.mean(axis=0)
            centered = psf_flat - psf_mean
            n_comp = min(self.n_components, centered.shape[0] - 1, centered.shape[1] - 1)
            _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
            components = vt[:n_comp]
            coeff = centered @ components.T
            coeff_mean = coeff.mean(axis=0)
            coeff_centered = coeff - coeff_mean
            dual_coeff = np.linalg.solve(dual_matrix, coeff_centered)
            weights = x_scaled.T @ dual_coeff

            total_var = float(np.sum(singular_values**2))
            evr = (singular_values[:n_comp] ** 2) / total_var if total_var > 0 else np.zeros(n_comp)

            self.components.append(components)
            self.explained_variance_ratio.append(evr)
            self.psf_means.append(psf_mean)
            self.coeff_means.append(coeff_mean)
            self.ridge_weights.append(weights)
            pred_coeff = x_scaled @ weights + coeff_mean
            ss_res = float(np.sum((coeff - pred_coeff) ** 2))
            ss_tot = float(np.sum((coeff - coeff.mean(axis=0)) ** 2)) + 1e-12
            fit_history[f"wl_{wl_idx}"] = {
                "explained_variance_ratio": evr.tolist(),
                "n_components_used": n_comp,
                "ridge_score": 1.0 - ss_res / ss_tot,
            }
        return fit_history

    def predict_coeff(self, masks: np.ndarray) -> list[np.ndarray]:
        x_scaled = self._scale_masks_transform(masks)
        return [x_scaled @ weights + coeff_mean for weights, coeff_mean in zip(self.ridge_weights, self.coeff_means)]

    def reconstruct(self, coeffs: list[np.ndarray]) -> np.ndarray:
        if self.psf_shape is None:
            raise RuntimeError("Model is not fitted")
        n_samples = coeffs[0].shape[0]
        height, width = self.psf_shape
        pred = np.zeros((n_samples, 1, self.n_wavelengths, height, width), dtype=np.float32)
        for wl_idx, coeff in enumerate(coeffs):
            psf_flat = coeff @ self.components[wl_idx] + self.psf_means[wl_idx]
            psf_flat = np.maximum(psf_flat, 0.0)
            psf_flat = psf_flat / (psf_flat.sum(axis=1, keepdims=True) + 1e-12)
            pred[:, 0, wl_idx] = psf_flat.reshape(n_samples, height, width)
        return pred


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _decode_path(path: Path | None) -> Path | None:
    return path.expanduser().resolve() if path is not None else None


def _resolve_dataset_paths(dataset: Path, test_dataset: Path | None) -> tuple[Path, Path | None, str]:
    dataset = _decode_path(dataset)
    test_dataset = _decode_path(test_dataset)
    if dataset is None:
        raise ValueError("--dataset is required")
    if test_dataset is not None:
        return dataset, test_dataset, "explicit_train_test"

    if dataset.is_dir():
        train_h5 = dataset / "train.h5"
        test_h5 = dataset / "test.h5"
        if train_h5.exists() and test_h5.exists():
            return train_h5, test_h5, "directory_train_test"
        raise FileNotFoundError(f"Directory dataset must contain train.h5 and test.h5: {dataset}")

    sibling_test = dataset.with_name("test.h5")
    if dataset.name == "train.h5" and sibling_test.exists():
        return dataset, sibling_test, "sibling_train_test"

    sibling_train = dataset.with_name("train.h5")
    if sibling_train.exists() and sibling_test.exists():
        return sibling_train, sibling_test, "sibling_train_test"

    return dataset, None, "single_h5_deterministic_split"


def _slice_data(data: dict[str, Any], indices: np.ndarray) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=int)
    sliced = dict(data)
    for key in ("masks", "psfs"):
        sliced[key] = data[key][indices]
    sliced["mask_id"] = [data["mask_id"][int(i)] for i in indices]
    sliced["mask_family"] = [data["mask_family"][int(i)] for i in indices]
    sliced["n_samples"] = len(indices)
    return sliced


def _load_train_test(
    dataset: Path,
    test_dataset: Path | None,
    psf_working_size: tuple[int, int] | None,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_path, resolved_test_path, mode = _resolve_dataset_paths(dataset, test_dataset)
    train_data = load_psf_dictionary(train_path, psf_working_size=psf_working_size)
    if resolved_test_path is not None:
        test_data = load_psf_dictionary(resolved_test_path, psf_working_size=psf_working_size)
        provenance = {
            "dataset_mode": mode,
            "train_h5": str(train_path),
            "test_h5": str(resolved_test_path),
        }
        return train_data, test_data, provenance

    all_data = train_data
    n_samples = int(all_data["n_samples"])
    if n_samples < 6:
        raise ValueError("Single-HDF5 deterministic split needs at least 6 samples")
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    n_test = max(2, int(round(n_samples * 0.2)))
    test_indices = np.sort(indices[:n_test])
    train_indices = np.sort(indices[n_test:])
    provenance = {
        "dataset_mode": mode,
        "train_h5": str(train_path),
        "test_h5": None,
        "split_seed": seed,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    return _slice_data(all_data, train_indices), _slice_data(all_data, test_indices), provenance


def _load_cached_artifacts(forward_artifacts: Path) -> dict[str, Any] | None:
    cache_path = forward_artifacts / "thesis_forward_cache.npz"
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    return {
        "cache_path": str(cache_path),
        "pca_components": data["pca_components"],
        "explained_variance_ratio": data["explained_variance_ratio"],
        "pred_psfs": data["pred_psfs"],
        "fit_history": data["fit_history"].item() if "fit_history" in data.files else {},
    }


def _fit_or_load_forward_model(
    train_data: dict[str, Any],
    test_data: dict[str, Any],
    forward_artifacts: Path,
    n_components: int,
    ridge_alpha: float,
    allow_recompute: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str, bool]:
    cached = _load_cached_artifacts(forward_artifacts)
    if cached is not None:
        return (
            cached["pca_components"],
            cached["explained_variance_ratio"],
            cached["pred_psfs"],
            cached.get("fit_history", {}),
            "artifact",
            False,
        )

    if not allow_recompute:
        raise FileNotFoundError(
            "Saved PCA/prediction arrays were not found under "
            f"{forward_artifacts}. Re-run with --allow-recompute to fit PCA+ridge "
            "deterministically from the measured PSF dictionary."
        )

    model = NumpyPCARidge(n_components=n_components, alpha=ridge_alpha)
    fit_history = model.fit(train_data["masks"], train_data["psfs"])
    pred_psfs = model.reconstruct(model.predict_coeff(test_data["masks"]))
    if model.psf_shape is None:
        raise RuntimeError("PCA+ridge model did not record PSF shape")
    height, width = model.psf_shape
    pca_components = [components.reshape(components.shape[0], height, width) for components in model.components]
    explained = model.explained_variance_ratio
    return (
        np.stack(pca_components, axis=0),
        np.stack(explained, axis=0),
        pred_psfs,
        fit_history,
        "recomputed",
        True,
    )


def _robust_symmetric_limit(image: np.ndarray, percentile: float = 99.0) -> float:
    limit = float(np.percentile(np.abs(image), percentile))
    return limit if limit > 0 else float(np.max(np.abs(image)) + 1e-12)


def _save_figure(fig: plt.Figure, out_base: Path, formats: str, dpi: int) -> dict[str, str]:
    outputs: dict[str, str] = {}
    if formats in {"png", "both"}:
        png_path = out_base.with_suffix(".png")
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        outputs["png"] = str(png_path)
    if formats in {"pdf", "both"}:
        pdf_path = out_base.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs["pdf"] = str(pdf_path)
    return outputs


def export_pca_basis_figure(
    components: np.ndarray,
    explained_variance: np.ndarray,
    wavelengths_nm: np.ndarray,
    out_dir: Path,
    num_pcs: int,
    formats: str,
    dpi: int,
) -> dict[str, str]:
    n_wavelengths = min(len(wavelengths_nm), components.shape[0])
    num_pcs = min(num_pcs, components.shape[1])
    fig, axes = plt.subplots(
        n_wavelengths,
        num_pcs,
        figsize=(2.35 * num_pcs, 2.15 * n_wavelengths),
        squeeze=False,
    )

    for wl_idx in range(n_wavelengths):
        for pc_idx in range(num_pcs):
            ax = axes[wl_idx, pc_idx]
            image = components[wl_idx, pc_idx]
            limit = _robust_symmetric_limit(image)
            ax.imshow(image, cmap="coolwarm", vmin=-limit, vmax=limit)
            ax.set_xticks([])
            ax.set_yticks([])
            if wl_idx == 0:
                evr = explained_variance[wl_idx, pc_idx] if pc_idx < explained_variance.shape[1] else np.nan
                ax.set_title(f"PC{pc_idx + 1}\nEVR {evr * 100:.1f}%", fontsize=9)
            if pc_idx == 0:
                ax.set_ylabel(f"{wavelengths_nm[wl_idx]:.0f} nm", fontsize=10, rotation=0, labelpad=30, va="center")

    fig.suptitle("Measured PSF PCA Basis Subset", fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0.9)
    return _save_figure(fig, out_dir / "fig4_pca_basis_subset", formats, dpi)


def _nearest_wavelength_indices(wavelengths_nm: np.ndarray) -> list[int]:
    indices = []
    for target in WAVELENGTH_RGB_ORDER:
        indices.append(int(np.argmin(np.abs(wavelengths_nm.astype(float) - target))))
    return indices


def _psf_pseudo_rgb(psf_lhw: np.ndarray, wavelengths_nm: np.ndarray, scale: float) -> np.ndarray:
    indices = _nearest_wavelength_indices(wavelengths_nm)
    rgb = np.stack([psf_lhw[idx] for idx in indices], axis=-1).astype(np.float64)
    scale = max(float(scale), 1e-12)
    rgb = np.clip(rgb / scale, 0.0, 1.0)
    rgb = np.log1p(12.0 * rgb) / math.log1p(12.0)
    return np.clip(rgb, 0.0, 1.0)


def _mask_thumbnail(mask: np.ndarray) -> np.ndarray:
    thumb = np.squeeze(mask)
    if thumb.ndim == 3:
        thumb = thumb[0]
    return thumb.astype(np.float64)


def _normalized_correlation(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_vec = gt.ravel().astype(np.float64)
    pred_vec = pred.ravel().astype(np.float64)
    gt_vec = gt_vec - gt_vec.mean()
    pred_vec = pred_vec - pred_vec.mean()
    denom = math.sqrt(float(np.sum(gt_vec * gt_vec) * np.sum(pred_vec * pred_vec))) + 1e-12
    return float(np.sum(gt_vec * pred_vec) / denom)


def _compute_sample_metrics(
    test_psfs: np.ndarray,
    pred_psfs: np.ndarray,
    wavelengths_nm: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    rows: list[dict[str, Any]] = []
    n_samples = test_psfs.shape[0]
    n_wavelengths = test_psfs.shape[2]
    corr_matrix = np.zeros((n_samples, n_wavelengths), dtype=np.float64)
    mean_corr = np.zeros(n_samples, dtype=np.float64)
    for sample_idx in range(n_samples):
        sample_corrs = []
        for wl_idx in range(n_wavelengths):
            gt = test_psfs[sample_idx, 0, wl_idx]
            pred = pred_psfs[sample_idx, 0, wl_idx]
            residual = np.abs(gt - pred)
            corr = _normalized_correlation(gt, pred)
            mse = float(np.mean((gt - pred) ** 2))
            mae = float(np.mean(residual))
            residual_l1 = float(np.sum(residual))
            corr_matrix[sample_idx, wl_idx] = corr
            sample_corrs.append(corr)
            rows.append(
                {
                    "sample_index": sample_idx,
                    "wavelength_nm": float(wavelengths_nm[wl_idx]),
                    "norm_corr": corr,
                    "mse": mse,
                    "mae": mae,
                    "residual_l1": residual_l1,
                }
            )
        mean_corr[sample_idx] = float(np.mean(sample_corrs))
    return rows, corr_matrix, mean_corr


def _mask_edge_energy(mask: np.ndarray) -> float:
    image = _mask_thumbnail(mask)
    gy, gx = np.gradient(image.astype(np.float64))
    return float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))


def _select_examples(
    test_data: dict[str, Any],
    mean_corr: np.ndarray,
    num_examples: int,
) -> tuple[list[int], str]:
    families = [str(x).lower() for x in test_data.get("mask_family", [])]
    selected: list[int] = []
    desired = ["deterministic", "lowfreq", "midfreq", "task"]
    for target in desired:
        candidates = [idx for idx, family in enumerate(families) if target in family and idx not in selected]
        if candidates:
            group_corr = mean_corr[candidates]
            median = float(np.median(group_corr))
            chosen = min(candidates, key=lambda idx: (abs(float(mean_corr[idx]) - median), idx))
            selected.append(chosen)
        if len(selected) >= num_examples:
            return selected[:num_examples], "family_balanced"

    if len(selected) >= min(num_examples, 2):
        policy = "family_balanced_with_quantile_fill"
    else:
        selected = []
        policy = "nc_quantile"

    remaining = [idx for idx in range(len(mean_corr)) if idx not in selected]
    if remaining:
        quantiles = [0.25, 0.50, 0.75]
        sorted_remaining = sorted(remaining, key=lambda idx: (mean_corr[idx], idx))
        for quantile in quantiles:
            if len(selected) >= num_examples:
                break
            pos = int(round(quantile * (len(sorted_remaining) - 1)))
            candidate = sorted_remaining[pos]
            if candidate not in selected:
                selected.append(candidate)

    if len(selected) < num_examples:
        edge_scores = [(_mask_edge_energy(test_data["masks"][idx]), idx) for idx in range(test_data["n_samples"]) if idx not in selected]
        edge_scores.sort(reverse=True)
        for _, idx in edge_scores:
            selected.append(idx)
            if len(selected) >= num_examples:
                break

    if len(selected) < num_examples:
        for idx in range(test_data["n_samples"]):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= num_examples:
                break

    return selected[:num_examples], policy


def write_metrics_csv(
    rows: list[dict[str, Any]],
    selected_indices: list[int],
    test_data: dict[str, Any],
    mean_corr: np.ndarray,
    out_path: Path,
) -> None:
    fieldnames = [
        "sample_id",
        "mask_family",
        "wavelength_nm",
        "norm_corr",
        "mse",
        "mae",
        "residual_l1",
        "display_note",
        "mean_norm_corr",
    ]
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_index.setdefault(int(row["sample_index"]), []).append(row)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_idx in selected_indices:
            sample_id = test_data["mask_id"][sample_idx]
            family = test_data["mask_family"][sample_idx]
            for row in by_index[sample_idx]:
                writer.writerow(
                    {
                        "sample_id": sample_id,
                        "mask_family": family,
                        "wavelength_nm": row["wavelength_nm"],
                        "norm_corr": row["norm_corr"],
                        "mse": row["mse"],
                        "mae": row["mae"],
                        "residual_l1": row["residual_l1"],
                        "display_note": "PSF pseudo-RGB uses shared measured/predicted log scale per sample; residual uses own log scale.",
                        "mean_norm_corr": "",
                    }
                )
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "mask_family": family,
                    "wavelength_nm": "mean",
                    "norm_corr": "",
                    "mse": "",
                    "mae": "",
                    "residual_l1": "",
                    "display_note": "per-sample mean over wavelengths",
                    "mean_norm_corr": float(mean_corr[sample_idx]),
                }
            )


def export_forward_prediction_figure(
    test_data: dict[str, Any],
    pred_psfs: np.ndarray,
    selected_indices: list[int],
    corr_matrix: np.ndarray,
    mean_corr: np.ndarray,
    out_dir: Path,
    formats: str,
    dpi: int,
) -> dict[str, str]:
    wavelengths_nm = test_data["wavelengths_nm"]
    test_psfs = test_data["psfs"]
    n_rows = len(selected_indices)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(12.2, 2.45 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.72, 1.0, 1.15, 1.15]},
    )
    column_titles = ["Sample", "Mask", "Measured", "Predicted"]

    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=FORWARD_COLUMN_TITLE_FONTSIZE, fontweight="bold")

    for row_idx, sample_idx in enumerate(selected_indices):
        mask = _mask_thumbnail(test_data["masks"][sample_idx])
        measured = test_psfs[sample_idx, 0]
        predicted = pred_psfs[sample_idx, 0]
        shared_scale = float(np.percentile(np.concatenate([measured.ravel(), predicted.ravel()]), 99.8))

        corr_text = " / ".join(f"{corr_matrix[sample_idx, wl_idx]:.3f}" for wl_idx in range(corr_matrix.shape[1]))
        sample_text = (
            f"{test_data['mask_id'][sample_idx]}\n"
            f"{test_data['mask_family'][sample_idx]}\n"
            f"mean NC={mean_corr[sample_idx]:.3f}\n"
            f"NC 450/550/650={corr_text}"
        )
        axes[row_idx, 0].text(
            0.0,
            0.5,
            sample_text,
            ha="left",
            va="center",
            fontsize=FORWARD_SAMPLE_TEXT_FONTSIZE,
            linespacing=1.28,
        )
        axes[row_idx, 0].set_axis_off()
        axes[row_idx, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row_idx, 2].imshow(_psf_pseudo_rgb(measured, wavelengths_nm, shared_scale))
        axes[row_idx, 3].imshow(_psf_pseudo_rgb(predicted, wavelengths_nm, shared_scale))

        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])

    fig.suptitle("Measured vs Predicted PSF Subset", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=0.85)
    outputs = {f"main_{key}": value for key, value in _save_figure(fig, out_dir / "fig4_forward_prediction_subset", formats, dpi).items()}

    residual_fig, residual_axes = plt.subplots(
        n_rows,
        3,
        figsize=(10.6, 2.55 * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.7, 1.0, 1.45]},
    )
    residual_titles = ["Sample", "Mask", "Residual"]
    for col_idx, title in enumerate(residual_titles):
        residual_axes[0, col_idx].set_title(title, fontsize=FORWARD_COLUMN_TITLE_FONTSIZE, fontweight="bold")

    for row_idx, sample_idx in enumerate(selected_indices):
        mask = _mask_thumbnail(test_data["masks"][sample_idx])
        measured = test_psfs[sample_idx, 0]
        predicted = pred_psfs[sample_idx, 0]
        residual = np.abs(measured - predicted)
        residual_scale = float(np.percentile(residual, 99.8))
        corr_text = " / ".join(f"{corr_matrix[sample_idx, wl_idx]:.3f}" for wl_idx in range(corr_matrix.shape[1]))
        sample_text = (
            f"{test_data['mask_id'][sample_idx]}\n"
            f"{test_data['mask_family'][sample_idx]}\n"
            f"mean NC={mean_corr[sample_idx]:.3f}\n"
            f"NC 450/550/650={corr_text}"
        )

        residual_axes[row_idx, 0].text(
            0.0,
            0.5,
            sample_text,
            ha="left",
            va="center",
            fontsize=FORWARD_SAMPLE_TEXT_FONTSIZE,
            linespacing=1.28,
        )
        residual_axes[row_idx, 0].set_axis_off()
        residual_axes[row_idx, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        residual_axes[row_idx, 2].imshow(_psf_pseudo_rgb(residual, wavelengths_nm, residual_scale))
        for col_idx in range(3):
            residual_axes[row_idx, col_idx].set_xticks([])
            residual_axes[row_idx, col_idx].set_yticks([])

    residual_fig.suptitle("Measured vs Predicted PSF Residuals", fontsize=14, fontweight="bold")
    residual_fig.tight_layout(pad=0.85)
    outputs.update(
        {f"residual_{key}": value for key, value in _save_figure(residual_fig, out_dir / "fig4_forward_prediction_residuals", formats, dpi).items()}
    )
    plt.close(fig)
    plt.close(residual_fig)
    return outputs


def _copy_to_thesis_assets(outputs: dict[str, str], copy_to: Path) -> list[str]:
    copy_to.mkdir(parents=True, exist_ok=True)
    copied = []
    for key in ("pca_pdf", "forward_pdf", "residual_pdf"):
        if key not in outputs:
            continue
        src = Path(outputs[key])
        dst = copy_to / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def export_figures(args: argparse.Namespace) -> dict[str, Any]:
    psf_working_size = tuple(args.psf_working_size) if args.psf_working_size else None
    train_data, test_data, data_provenance = _load_train_test(
        args.dataset,
        args.test_dataset,
        psf_working_size,
        args.seed,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    components, explained, pred_psfs, fit_history, pca_source, pca_recomputed = _fit_or_load_forward_model(
        train_data,
        test_data,
        args.forward_artifacts,
        args.num_components_total,
        args.ridge_alpha,
        args.allow_recompute,
    )
    metric_rows, corr_matrix, mean_corr = _compute_sample_metrics(test_data["psfs"], pred_psfs, test_data["wavelengths_nm"])
    selected_indices, selection_policy = _select_examples(test_data, mean_corr, args.num_examples)

    pca_outputs = export_pca_basis_figure(
        components,
        explained,
        test_data["wavelengths_nm"],
        args.out_dir,
        args.num_pcs,
        args.format,
        args.dpi,
    )
    forward_outputs = export_forward_prediction_figure(
        test_data,
        pred_psfs,
        selected_indices,
        corr_matrix,
        mean_corr,
        args.out_dir,
        args.format,
        args.dpi,
    )
    metrics_csv = args.out_dir / "fig4_forward_prediction_subset_metrics.csv"
    write_metrics_csv(metric_rows, selected_indices, test_data, mean_corr, metrics_csv)

    outputs = {
        "pca_pdf": str(args.out_dir / "fig4_pca_basis_subset.pdf"),
        "pca_png": str(args.out_dir / "fig4_pca_basis_subset.png"),
        "forward_pdf": forward_outputs.get("main_pdf", str(args.out_dir / "fig4_forward_prediction_subset.pdf")),
        "forward_png": forward_outputs.get("main_png", str(args.out_dir / "fig4_forward_prediction_subset.png")),
        "residual_pdf": forward_outputs.get("residual_pdf", str(args.out_dir / "fig4_forward_prediction_residuals.pdf")),
        "residual_png": forward_outputs.get("residual_png", str(args.out_dir / "fig4_forward_prediction_residuals.png")),
        "metrics_csv": str(metrics_csv),
    }

    copied = []
    if args.copy_to_thesis_assets:
        copied = _copy_to_thesis_assets(outputs, args.copy_to_thesis_assets)

    selected_sample_ids = [test_data["mask_id"][idx] for idx in selected_indices]
    selected_families = [test_data["mask_family"][idx] for idx in selected_indices]
    manifest = {
        "task": "thesis_forward_figures",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": data_provenance,
        "wavelengths_nm": [float(x) for x in test_data["wavelengths_nm"]],
        "psf_shape": list(test_data["psf_shape"]),
        "display_shape": (
            "native PSF shape; main figure shows mask/measured/predicted with shared measured/predicted "
            "log scale per sample; companion residual figure uses a separate residual log scale"
        ),
        "pca_recomputed": pca_recomputed,
        "pca_basis": {
            "figure": "fig4_pca_basis_subset",
            "wavelengths_nm": [float(x) for x in test_data["wavelengths_nm"]],
            "source": pca_source,
            "num_components_total": int(components.shape[1]),
            "num_components_shown": int(min(args.num_pcs, components.shape[1])),
            "normalization": "per-component symmetric p99",
            "input_dataset": data_provenance["train_h5"],
            "output_pdf": outputs["pca_pdf"],
            "fit_history": fit_history,
        },
        "prediction_subset": {
            "selection_policy": selection_policy,
            "num_examples": len(selected_indices),
            "sample_ids": selected_sample_ids,
            "mask_families": selected_families,
            "mean_norm_corr": float(np.mean(mean_corr[selected_indices])),
            "per_sample_mean_norm_corr": {
                test_data["mask_id"][idx]: float(mean_corr[idx]) for idx in selected_indices
            },
        },
        "outputs": outputs,
        "copied_to_thesis_assets": copied,
        "figure_generation": {
            "dpi": args.dpi,
            "format": args.format,
            "ridge_alpha": args.ridge_alpha,
            "seed": args.seed,
        },
    }
    manifest_path = args.out_dir / "thesis_figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"Wrote {outputs['pca_pdf']}")
    print(f"Wrote {outputs['forward_pdf']}")
    print(f"Wrote {outputs['residual_pdf']}")
    print(f"Wrote {metrics_csv}")
    print(f"Wrote {manifest_path}")
    for path in copied:
        print(f"Copied {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export thesis-ready Phase 3.5 forward-model figures")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="Train HDF5, directory with train/test HDF5s, or single HDF5 to split deterministically.")
    parser.add_argument("--test-dataset", type=Path, default=None, help="Optional explicit held-out test HDF5.")
    parser.add_argument("--forward-artifacts", type=Path, default=DEFAULT_FORWARD_ARTIFACTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-pcs", type=int, default=3)
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--num-components-total", type=int, default=24)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--allow-recompute", action="store_true")
    parser.add_argument("--copy-to-thesis-assets", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--format", choices=["png", "pdf", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--psf-working-size", nargs=2, type=int, default=None, metavar=("H", "W"))
    args = parser.parse_args()
    export_figures(args)


if __name__ == "__main__":
    main()
