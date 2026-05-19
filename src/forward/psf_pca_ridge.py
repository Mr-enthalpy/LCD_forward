from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


class PSFPCARidge:
    def __init__(
        self,
        n_components: int = 24,
        alpha: float = 1.0,
        per_wavelength: bool = True,
        mask_scale: bool = True,
    ):
        self.n_components = n_components
        self.alpha = alpha
        self.per_wavelength = per_wavelength
        self.mask_scale = mask_scale

        self.pca_models: Dict[int, PCA] = {}
        self.ridge_models: Dict[int, Ridge] = {}
        self.mask_scalers: Dict[int, StandardScaler] = {}
        self.psf_means: Dict[int, np.ndarray] = {}
        self.psf_shapes: Dict[int, Tuple[int, int]] = {}
        self.n_wavelengths: int = 0
        self.fitted: bool = False

    def fit(
        self,
        train_masks: np.ndarray,
        train_psfs: np.ndarray,
        wavelengths_nm: Optional[np.ndarray] = None,
    ) -> Dict:
        n_samples = train_masks.shape[0]
        n_wl = train_psfs.shape[2] if train_psfs.ndim >= 3 else 1
        self.n_wavelengths = n_wl

        masks_flat = train_masks.reshape(n_samples, -1).astype(np.float64)
        psf_h, psf_w = train_psfs.shape[-2:]
        psf_dim = psf_h * psf_w

        fit_history = {"n_components": self.n_components, "alpha": self.alpha}

        for wl_idx in range(n_wl):
            psf_slice = train_psfs[:, 0, wl_idx, :, :].reshape(n_samples, psf_dim).astype(np.float64)
            psf_mean = psf_slice.mean(axis=0)
            psf_centered = psf_slice - psf_mean

            n_comp = min(self.n_components, n_samples - 1, psf_dim - 1)
            pca = PCA(n_components=n_comp, random_state=42)
            coeff = pca.fit_transform(psf_centered)

            self.pca_models[wl_idx] = pca
            self.psf_means[wl_idx] = psf_mean
            self.psf_shapes[wl_idx] = (psf_h, psf_w)

            scaler = StandardScaler()
            masks_scaled = scaler.fit_transform(masks_flat)

            ridge = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge.fit(masks_scaled, coeff)

            self.ridge_models[wl_idx] = ridge
            self.mask_scalers[wl_idx] = scaler

            fit_history[f"wl_{wl_idx}"] = {
                "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                "n_components_used": n_comp,
                "ridge_score": float(ridge.score(masks_scaled, coeff)),
            }

        self.fitted = True
        return fit_history

    def predict_coeff(self, masks: np.ndarray) -> Dict[int, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples = masks.shape[0]
        masks_flat = masks.reshape(n_samples, -1).astype(np.float64)

        coeffs = {}
        for wl_idx in range(self.n_wavelengths):
            masks_scaled = self.mask_scalers[wl_idx].transform(masks_flat)
            coeffs[wl_idx] = self.ridge_models[wl_idx].predict(masks_scaled)
        return coeffs

    def reconstruct(self, coeffs: Dict[int, np.ndarray]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model not fitted.")

        n_samples = list(coeffs.values())[0].shape[0]
        psf_h, psf_w = self.psf_shapes[0]

        psfs_pred = np.zeros((n_samples, 1, self.n_wavelengths, psf_h, psf_w), dtype=np.float32)

        for wl_idx in range(self.n_wavelengths):
            coeff = coeffs[wl_idx]
            pca = self.pca_models[wl_idx]
            psf_mean = self.psf_means[wl_idx]

            psf_flat = pca.inverse_transform(coeff) + psf_mean
            psf_flat = np.maximum(psf_flat, 0.0)
            psf_flat = psf_flat / (psf_flat.sum(axis=1, keepdims=True) + 1e-8)
            psfs_pred[:, 0, wl_idx] = psf_flat.reshape(n_samples, psf_h, psf_w)

        return psfs_pred.astype(np.float32)

    def fit_predict(
        self,
        train_masks: np.ndarray,
        train_psfs: np.ndarray,
        test_masks: np.ndarray,
    ) -> np.ndarray:
        self.fit(train_masks, train_psfs)
        coeffs = self.predict_coeff(test_masks)
        return self.reconstruct(coeffs)


def compute_psf_metrics(
    gt_psfs: np.ndarray,
    pred_psfs: np.ndarray,
) -> Dict:
    gt = gt_psfs.astype(np.float64)
    pred = pred_psfs.astype(np.float64)

    n_wl = gt.shape[2] if gt.ndim >= 3 else 1

    metrics = {"per_wavelength": {}, "mean": {}}

    for wl_idx in range(n_wl):
        gt_wl = gt[:, 0, wl_idx].ravel() if gt.ndim >= 5 else gt.ravel()
        pred_wl = pred[:, 0, wl_idx].ravel() if pred.ndim >= 5 else pred.ravel()

        mse = float(np.mean((gt_wl - pred_wl) ** 2))
        rel_l2 = float(np.sqrt(np.sum((gt_wl - pred_wl) ** 2)) / (np.sqrt(np.sum(gt_wl ** 2)) + 1e-8))

        gt_c = gt_wl - gt_wl.mean()
        pred_c = pred_wl - pred_wl.mean()
        denom = np.sqrt(np.sum(gt_c ** 2) * np.sum(pred_c ** 2)) + 1e-8
        corr = float(np.sum(gt_c * pred_c) / denom)

        metrics["per_wavelength"][str(wl_idx)] = {
            "mse": mse,
            "relative_l2": rel_l2,
            "normalized_correlation": corr,
        }

    all_mse = [metrics["per_wavelength"][str(wl)]["mse"] for wl in range(n_wl)]
    all_rel = [metrics["per_wavelength"][str(wl)]["relative_l2"] for wl in range(n_wl)]
    all_corr = [metrics["per_wavelength"][str(wl)]["normalized_correlation"] for wl in range(n_wl)]

    metrics["mean"] = {
        "mse": float(np.mean(all_mse)),
        "relative_l2": float(np.mean(all_rel)),
        "normalized_correlation": float(np.mean(all_corr)),
    }

    return metrics
