from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import torch


def _decode_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_scalar(value.item())
        return [_decode_scalar(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _decode_scalar(value.item())
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_dataset(group: h5py.Group, name: str) -> Any:
    if name not in group:
        raise KeyError(f"Missing required HDF5 dataset: {group.name}/{name}")
    return _decode_scalar(group[name][()])


def _center_crop(images: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    height, width = images.shape[-2:]
    target_h, target_w = target_shape
    if target_h > height or target_w > width:
        raise ValueError(
            f"Cannot center-crop residuals from {(height, width)} to larger target {target_shape}"
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return images[..., top:top + target_h, left:left + target_w]


def _as_jsonable_indices(indices: Iterable[int], max_items: int = 256) -> dict[str, Any]:
    values = [int(i) for i in indices]
    preview = values[:max_items]
    return {
        "count": len(values),
        "preview": preview,
        "truncated": len(values) > len(preview),
    }


class ClosedLCDResidualNoise:
    def __init__(
        self,
        h5_path: str | Path,
        target_shape: tuple[int, int],
        sample_policy: str = "pooled",
        count_peak: float = 200.0,
        quantile: float = 0.999,
        resize_mode: str = "center_crop",
        seed: int = 0,
    ):
        self.h5_path = Path(h5_path)
        self.target_shape = tuple(int(x) for x in target_shape)
        self.sample_policy = sample_policy
        self.count_peak = float(count_peak)
        self.quantile = float(quantile)
        self.resize_mode = resize_mode
        self.seed = int(seed)
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

        if self.sample_policy not in {"pooled", "cycle_by_frame", "per_lambda_weighted"}:
            raise ValueError(f"Unsupported residual sample_policy: {self.sample_policy}")
        if self.count_peak <= 0:
            raise ValueError("count_peak must be positive")
        if not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")

        self.residual_bank, self.release_metadata = self._load_release()
        self.residual_transform = self._describe_transform()

    @classmethod
    def from_config(
        cls,
        cfg: dict[str, Any],
        target_shape: tuple[int, int],
    ) -> "ClosedLCDResidualNoise":
        return cls(
            h5_path=cfg["source_h5"],
            target_shape=target_shape,
            sample_policy=cfg.get("sample_policy", "pooled"),
            count_peak=cfg.get("count_peak", 200.0),
            quantile=cfg.get("scale_quantile", cfg.get("quantile", 0.999)),
            resize_mode=cfg.get("resize_mode", "center_crop"),
            seed=cfg.get("seed", 0),
        )

    def _load_release(self) -> tuple[np.ndarray, dict[str, Any]]:
        if not self.h5_path.exists():
            raise FileNotFoundError(f"Closed-LCD residual HDF5 not found: {self.h5_path}")

        with h5py.File(self.h5_path, "r") as handle:
            if "closed_lcd" not in handle:
                raise KeyError("Missing HDF5 group: /closed_lcd")
            if "metadata" not in handle:
                raise KeyError("Missing HDF5 group: /metadata")

            residuals = np.asarray(handle["closed_lcd/residuals_avg10"], dtype=np.float32)
            metadata_group = handle["metadata"]
            metadata = {
                "wavelengths_nm": _read_dataset(metadata_group, "wavelengths_nm"),
                "exposure_us": _read_dataset(metadata_group, "exposure_us"),
                "n_avg_frames": int(_read_dataset(metadata_group, "n_avg_frames")),
                "source_mask_id": str(_read_dataset(metadata_group, "source_mask_id")),
                "roi_name": str(_read_dataset(metadata_group, "roi_name")),
                "roi_shape": _read_dataset(metadata_group, "roi_shape"),
                "is_closed_lcd_residual": bool(_read_dataset(metadata_group, "is_closed_lcd_residual")),
                "is_sensor_dark": bool(_read_dataset(metadata_group, "is_sensor_dark")),
                "is_single_frame_burst": bool(_read_dataset(metadata_group, "is_single_frame_burst")),
            }
            if "n_repeats" in metadata_group:
                metadata["n_repeats"] = int(_read_dataset(metadata_group, "n_repeats"))
            if "source_psf_dictionary" in metadata_group:
                metadata["source_psf_dictionary"] = str(_read_dataset(metadata_group, "source_psf_dictionary"))

        if residuals.ndim != 4:
            raise ValueError(f"Expected residuals_avg10 [L,R,H,W], got {residuals.shape}")
        if metadata["n_avg_frames"] != 10:
            raise ValueError(f"Expected n_avg_frames == 10, got {metadata['n_avg_frames']}")
        if metadata["source_mask_id"] != "all_closed_window":
            raise ValueError(
                "Closed-LCD residual release must have source_mask_id='all_closed_window'; "
                f"got {metadata['source_mask_id']!r}"
            )
        if not metadata["is_closed_lcd_residual"]:
            raise ValueError("Release is not marked as closed-LCD residual data")
        if metadata["is_sensor_dark"]:
            raise ValueError("Closed-LCD residual release must not be marked as sensor dark")
        if metadata["is_single_frame_burst"]:
            raise ValueError("Closed-LCD residual release must not be marked as single-frame burst")

        residuals = self._transform_residuals(residuals)
        return residuals, metadata

    def _transform_residuals(self, residuals: np.ndarray) -> np.ndarray:
        if self.resize_mode == "center_crop":
            residuals = _center_crop(residuals, self.target_shape)
        elif self.resize_mode == "none":
            if tuple(residuals.shape[-2:]) != self.target_shape:
                raise ValueError(
                    f"resize_mode='none' requires residual shape {self.target_shape}, "
                    f"got {tuple(residuals.shape[-2:])}"
                )
        else:
            raise ValueError(f"Unsupported resize_mode: {self.resize_mode}")

        if tuple(residuals.shape[-2:]) != self.target_shape:
            raise ValueError(
                f"Residual transform produced {tuple(residuals.shape[-2:])}, expected {self.target_shape}"
            )
        return np.ascontiguousarray(residuals.astype(np.float32))

    def _describe_transform(self) -> str:
        if self.resize_mode == "center_crop":
            return f"center_crop_to_{self.target_shape[0]}x{self.target_shape[1]}"
        return self.resize_mode

    @property
    def residuals_flat(self) -> np.ndarray:
        return self.residual_bank.reshape(-1, *self.residual_bank.shape[-2:])

    def _sample_flat_indices(self, shape: tuple[int, ...]) -> np.ndarray:
        bank_len = self.residuals_flat.shape[0]
        if self.sample_policy == "pooled":
            total = int(np.prod(shape))
            indices = torch.randint(bank_len, (total,), generator=self._generator).numpy()
            return indices.reshape(shape)
        if self.sample_policy == "cycle_by_frame":
            frame_count = shape[-1]
            frame_indices = np.arange(frame_count, dtype=np.int64) % bank_len
            return np.broadcast_to(frame_indices, shape).copy()
        raise ValueError("per_lambda_weighted does not use flat residual indices")

    def _sample_residuals(self, frame_shape: tuple[int, ...]) -> tuple[torch.Tensor, dict[str, Any]]:
        target_h, target_w = self.target_shape
        if self.sample_policy in {"pooled", "cycle_by_frame"}:
            flat_indices = self._sample_flat_indices(frame_shape)
            residuals = self.residuals_flat[flat_indices.reshape(-1)]
            residuals = residuals.reshape(*frame_shape, target_h, target_w)
            return torch.from_numpy(residuals), {
                "sampled_flat_indices": _as_jsonable_indices(flat_indices.reshape(-1)),
            }

        n_lambda, n_repeats, _, _ = self.residual_bank.shape
        total = int(np.prod(frame_shape))
        repeat_indices = torch.randint(n_repeats, (total,), generator=self._generator).numpy()
        residuals = self.residual_bank[:, repeat_indices].mean(axis=0)
        residuals = residuals.reshape(*frame_shape, target_h, target_w)
        return torch.from_numpy(residuals), {
            "sampled_repeat_indices": _as_jsonable_indices(repeat_indices),
            "per_lambda_weighted_note": (
                "Uniform wavelength average of residual bank; not recommended for main results."
            ),
            "n_wavelengths_averaged": int(n_lambda),
        }

    def apply(self, frames_clean: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        if not isinstance(frames_clean, torch.Tensor):
            raise TypeError("frames_clean must be a torch.Tensor")
        if tuple(frames_clean.shape[-2:]) != self.target_shape:
            raise ValueError(
                f"frames_clean spatial shape {tuple(frames_clean.shape[-2:])} does not match "
                f"target_shape {self.target_shape}"
            )

        original_shape = tuple(frames_clean.shape)
        if frames_clean.ndim == 3:
            frame_shape = (frames_clean.shape[0],)
            expand = "thw"
        elif frames_clean.ndim == 5 and frames_clean.shape[2] == 1:
            frame_shape = (frames_clean.shape[0], frames_clean.shape[1])
            expand = "ntchw"
        else:
            raise ValueError(
                "frames_clean must be [T,H,W] or [N,T,1,H,W], "
                f"got shape {original_shape}"
            )

        clean_for_scale = frames_clean.detach().float().cpu()
        clean_quantile = float(torch.quantile(clean_for_scale.reshape(-1), self.quantile))
        safe_quantile = max(clean_quantile, 1e-12)
        gain = self.count_peak / safe_quantile

        residuals, sample_meta = self._sample_residuals(frame_shape)
        if expand == "ntchw":
            residuals = residuals.unsqueeze(2)
        residuals = residuals.to(device=frames_clean.device, dtype=frames_clean.dtype)
        frames_noisy = frames_clean + residuals / gain

        metadata = {
            "enabled": True,
            "type": "closed_lcd_avg10_residual",
            "setting": "closed_lcd_residual",
            "source_h5": str(self.h5_path),
            "source_mask_id": self.release_metadata["source_mask_id"],
            "n_avg_frames": self.release_metadata["n_avg_frames"],
            "is_sensor_dark": self.release_metadata["is_sensor_dark"],
            "is_single_frame_burst": self.release_metadata["is_single_frame_burst"],
            "count_peak": self.count_peak,
            "scale_quantile": self.quantile,
            "clean_quantile": clean_quantile,
            "gain_counts_per_normalized_unit": gain,
            "sample_policy": self.sample_policy,
            "resize_mode": self.resize_mode,
            "residual_transform": self.residual_transform,
            "target_shape": list(self.target_shape),
            "input_shape": list(original_shape),
            "residual_bank_shape": list(self.residual_bank.shape),
            "residual_mean_count": float(np.mean(self.residual_bank)),
            "residual_std_count": float(np.std(self.residual_bank)),
            "seed": self.seed,
            "claim_level": "averaged_frame_closed_lcd_residual",
            "forbidden_interpretation": (
                "not PSF noise, not forward-surrogate training data, not sensor/read/shot/PRNU noise model"
            ),
        }
        metadata.update(sample_meta)
        return frames_noisy, metadata


def apply_closed_lcd_residual_noise(
    frames_clean: torch.Tensor,
    cfg: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if not cfg or not cfg.get("enabled", False):
        return frames_clean, {"enabled": False, "setting": "clean"}
    if cfg.get("type") != "closed_lcd_avg10_residual":
        raise ValueError(f"Unsupported add_noise.type: {cfg.get('type')}")
    model = ClosedLCDResidualNoise.from_config(cfg, target_shape=tuple(frames_clean.shape[-2:]))
    return model.apply(frames_clean)
