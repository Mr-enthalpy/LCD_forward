from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_thesis_forward_figures import export_figures


def _write_psf_h5(path: Path, n_samples: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    mask_size = 8
    psf_size = 16
    wavelengths = np.array([450.0, 550.0, 650.0], dtype=np.float32)
    families = ["deterministic", "lowfreq", "midfreq", "task"]

    masks = np.zeros((n_samples, 1, 1, mask_size, mask_size), dtype=np.uint8)
    psfs = np.zeros((n_samples, 1, 3, psf_size, psf_size), dtype=np.float32)
    yy, xx = np.mgrid[:psf_size, :psf_size]

    for idx in range(n_samples):
        mask = rng.integers(0, 2, size=(mask_size, mask_size), dtype=np.uint8) * 255
        masks[idx, 0, 0] = mask
        center_y = psf_size / 2 + (mask.mean() / 255.0 - 0.5) * 2.0
        center_x = psf_size / 2 + ((mask[:, : mask_size // 2].mean() - mask[:, mask_size // 2 :].mean()) / 255.0)
        for wl_idx in range(3):
            sigma = 1.8 + wl_idx * 0.45 + 0.05 * (idx % 3)
            blob = np.exp(-((yy - center_y - wl_idx * 0.2) ** 2 + (xx - center_x + wl_idx * 0.15) ** 2) / (2 * sigma**2))
            ripple = 0.05 * np.sin((idx + 1) * xx / psf_size * np.pi) ** 2
            psf = np.maximum(blob + ripple, 0.0)
            psfs[idx, 0, wl_idx] = psf / psf.sum()

    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        handle.create_dataset("masks", data=masks)
        handle.create_dataset("psfs", data=psfs)
        handle.create_dataset("wavelengths_nm", data=wavelengths)
        handle.create_dataset("mask_id", data=np.array([f"mask_{idx:03d}" for idx in range(n_samples)], dtype=object), dtype=str_dtype)
        handle.create_dataset("mask_family", data=np.array([families[idx % len(families)] for idx in range(n_samples)], dtype=object), dtype=str_dtype)
        handle.create_dataset("metadata_json", data=json.dumps({"fixture": True}), dtype=str_dtype)


def test_export_thesis_forward_figures_smoke(tmp_path: Path) -> None:
    train_h5 = tmp_path / "train.h5"
    test_h5 = tmp_path / "test.h5"
    artifacts = tmp_path / "forward_artifacts"
    out_dir = tmp_path / "thesis_figures"
    _write_psf_h5(train_h5, n_samples=10, seed=1)
    _write_psf_h5(test_h5, n_samples=6, seed=2)
    artifacts.mkdir()

    args = SimpleNamespace(
        dataset=train_h5,
        test_dataset=test_h5,
        forward_artifacts=artifacts,
        out_dir=out_dir,
        num_pcs=3,
        num_examples=4,
        num_components_total=5,
        ridge_alpha=1.0,
        allow_recompute=True,
        copy_to_thesis_assets=None,
        dpi=80,
        format="both",
        seed=42,
        psf_working_size=None,
    )

    export_figures(args)

    expected = [
        out_dir / "fig4_pca_basis_subset.pdf",
        out_dir / "fig4_pca_basis_subset.png",
        out_dir / "fig4_forward_prediction_subset.pdf",
        out_dir / "fig4_forward_prediction_subset.png",
        out_dir / "fig4_forward_prediction_subset_metrics.csv",
        out_dir / "thesis_figures_manifest.json",
    ]
    for path in expected:
        assert path.exists()
        assert path.stat().st_size > 0

    with (out_dir / "fig4_forward_prediction_subset_metrics.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert {"sample_id", "mask_family", "wavelength_nm", "norm_corr", "mse", "mae", "residual_l1", "display_note"}.issubset(reader.fieldnames or [])
        rows = list(reader)
        assert any(row["wavelength_nm"] == "mean" for row in rows)

    manifest = json.loads((out_dir / "thesis_figures_manifest.json").read_text(encoding="utf-8"))
    assert manifest["task"] == "thesis_forward_figures"
    assert manifest["pca_recomputed"] is True
    assert len(manifest["prediction_subset"]["sample_ids"]) == 4
    assert manifest["prediction_subset"]["mean_norm_corr"] > 0.0
