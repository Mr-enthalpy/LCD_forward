from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.noise.closed_lcd_residual import (
    ClosedLCDResidualNoise,
    apply_closed_lcd_residual_noise,
)


def _write_release(path: Path, source_mask_id: str = "all_closed_window") -> None:
    rng = np.random.default_rng(0)
    residuals = rng.normal(0.0, 0.01, size=(3, 2, 8, 8)).astype(np.float32)
    residuals -= residuals.mean(dtype=np.float64).astype(np.float32)
    with h5py.File(path, "w") as handle:
        closed = handle.create_group("closed_lcd")
        closed.create_dataset("residuals_avg10", data=residuals)
        closed.create_dataset("mean_avg10", data=np.full((3, 8, 8), 25.0, dtype=np.float32))
        metadata = handle.create_group("metadata")
        metadata.create_dataset("wavelengths_nm", data=np.array([450.0, 550.0, 650.0]))
        metadata.create_dataset("exposure_us", data=np.array([1000.0, 1000.0, 1000.0]))
        metadata.create_dataset("n_avg_frames", data=np.int32(10))
        metadata.create_dataset("n_repeats", data=np.int32(2))
        metadata.create_dataset("source_mask_id", data=source_mask_id)
        metadata.create_dataset("roi_name", data="roi_512")
        metadata.create_dataset("roi_shape", data=np.array([8, 8]))
        metadata.create_dataset("is_closed_lcd_residual", data=True)
        metadata.create_dataset("is_sensor_dark", data=False)
        metadata.create_dataset("is_single_frame_burst", data=False)


def test_loads_release_and_validates_metadata(tmp_path: Path) -> None:
    h5_path = tmp_path / "closed_lcd_roi512_avg10_residuals.h5"
    _write_release(h5_path)

    noise = ClosedLCDResidualNoise(
        h5_path,
        target_shape=(4, 4),
        sample_policy="pooled",
        resize_mode="center_crop",
        seed=7,
    )

    assert noise.residual_bank.shape == (3, 2, 4, 4)
    assert noise.release_metadata["n_avg_frames"] == 10
    assert noise.release_metadata["is_sensor_dark"] is False
    assert noise.release_metadata["source_mask_id"] == "all_closed_window"
    assert abs(float(noise.residual_bank.mean())) < 1e-2


def test_apply_preserves_shape_and_is_reproducible(tmp_path: Path) -> None:
    h5_path = tmp_path / "closed_lcd_roi512_avg10_residuals.h5"
    _write_release(h5_path)
    frames = torch.ones(2, 3, 1, 4, 4, dtype=torch.float32)

    cfg = {
        "enabled": True,
        "type": "closed_lcd_avg10_residual",
        "source_h5": str(h5_path),
        "sample_policy": "pooled",
        "count_peak": 200.0,
        "scale_quantile": 0.999,
        "resize_mode": "center_crop",
        "seed": 11,
    }
    noisy_a, meta_a = apply_closed_lcd_residual_noise(frames, cfg)
    noisy_b, meta_b = apply_closed_lcd_residual_noise(frames, cfg)

    assert noisy_a.shape == frames.shape
    assert torch.allclose(noisy_a, noisy_b)
    assert meta_a["sampled_flat_indices"] == meta_b["sampled_flat_indices"]
    assert meta_a["gain_counts_per_normalized_unit"] > 0
    assert meta_a["setting"] == "closed_lcd_residual"


def test_disabled_noise_returns_clean_frames() -> None:
    frames = torch.rand(3, 4, 4)
    output, metadata = apply_closed_lcd_residual_noise(frames, {"enabled": False})
    assert output is frames
    assert metadata == {"enabled": False, "setting": "clean"}


def test_rejects_non_closed_window_release(tmp_path: Path) -> None:
    h5_path = tmp_path / "bad_residuals.h5"
    _write_release(h5_path, source_mask_id="ordinary_mask")

    with pytest.raises(ValueError, match="all_closed_window"):
        ClosedLCDResidualNoise(h5_path, target_shape=(4, 4))
