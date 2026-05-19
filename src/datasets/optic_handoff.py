from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]


def _resolve_env_var(value: str) -> str:
    if value.startswith("${") and "}" in value:
        inner = value[2:]
        inner = inner.rstrip("}")
        if ":" in inner:
            var, default = inner.split(":", 1)
        else:
            var, default = inner, ""
        import os
        return os.environ.get(var.strip(), default.strip())
    return value


def _decode_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_psf_dictionary(
    h5_path: PathLike,
    psf_working_size: Optional[Tuple[int, int]] = None,
    normalize_masks: bool = True,
) -> Dict:
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        masks = np.asarray(f["masks"]).astype(np.float32)
        psfs = np.asarray(f["psfs"]).astype(np.float32)
        wavelengths_nm = np.asarray(f["wavelengths_nm"]).astype(np.float32)

        mask_ids_raw = f["mask_id"][:]
        mask_ids = [_decode_str(x) for x in mask_ids_raw]

        mask_families_raw = f["mask_family"][:]
        mask_families = [_decode_str(x) for x in mask_families_raw]

        meta_raw = f["metadata_json"][()]
        metadata_json = json.loads(_decode_str(meta_raw))

    if normalize_masks:
        masks = masks / 255.0

    if psf_working_size is not None and tuple(psf_working_size) != tuple(psfs.shape[-2:]):
        psfs = _resize_psfs(psfs, psf_working_size)

    psfs = _ensure_nonnegative(psfs)

    return {
        "masks": masks,
        "psfs": psfs,
        "wavelengths_nm": wavelengths_nm,
        "mask_id": mask_ids,
        "mask_family": mask_families,
        "metadata_json": metadata_json,
        "n_samples": masks.shape[0],
        "mask_shape": masks.shape[-2:],
        "psf_shape": psfs.shape[-2:],
        "n_wavelengths": len(wavelengths_nm),
    }


def _resize_psfs(psfs: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    from scipy.ndimage import zoom

    n, t, l, h_orig, w_orig = psfs.shape
    h_new, w_new = target_size
    zoom_h = h_new / h_orig
    zoom_w = w_new / w_orig

    resized = np.zeros((n, t, l, h_new, w_new), dtype=np.float32)
    for i in range(n):
        for j in range(t):
            for k in range(l):
                resized[i, j, k] = zoom(psfs[i, j, k], (zoom_h, zoom_w), order=1)

    sums = resized.sum(axis=(-2, -1), keepdims=True)
    resized = resized / (sums + 1e-8)
    return resized.astype(np.float32)


def _ensure_nonnegative(psfs: np.ndarray) -> np.ndarray:
    return np.maximum(psfs, 0.0)


def get_psfs_by_wavelength(data: Dict, wavelength_idx: int) -> np.ndarray:
    psfs = data["psfs"]
    if psfs.ndim == 5:
        return psfs[..., wavelength_idx, :, :]
    return psfs


def get_masks_flat(data: Dict) -> np.ndarray:
    masks = data["masks"]
    if masks.ndim == 5:
        n, t, c, hm, wm = masks.shape
        return masks.reshape(n, -1)
    elif masks.ndim == 4:
        return masks.reshape(masks.shape[0], -1)
    return masks


def select_masks_by_strategy(
    data: Dict,
    strategy: str = "representative_first",
    count: int = 9,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    mask_ids = data["mask_id"]
    mask_families = data["mask_family"]
    n = data["n_samples"]

    if strategy == "representative_first":
        seen_families = {}
        indices = []
        for i in range(n):
            fam = mask_families[i]
            if fam not in seen_families:
                seen_families[fam] = True
                indices.append(i)
                if len(indices) >= count:
                    break
        if len(indices) < count:
            remaining = [i for i in range(n) if i not in indices]
            indices.extend(remaining[: count - len(indices)])
    elif strategy == "first_n":
        indices = list(range(min(count, n)))
    else:
        indices = list(range(min(count, n)))

    selected_masks = data["masks"][indices]
    selected_psfs = data["psfs"][indices]
    selected_ids = [mask_ids[i] for i in indices]
    selected_families = [mask_families[i] for i in indices]

    return selected_masks, selected_psfs, selected_ids, selected_families
