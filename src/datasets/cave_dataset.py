from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]


def _decode_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def load_cave_dataset(
    h5_path: PathLike,
    indices: Optional[List[int]] = None,
) -> Dict:
    h5_path = str(h5_path)
    with h5py.File(h5_path, "r") as f:
        objects = np.asarray(f["objects"]).astype(np.float32)
        wavelengths_nm = np.asarray(f["wavelengths_nm"]).astype(np.float32)
        scene_ids_raw = f["scene_id"][:]
        scene_ids = [_decode_str(x) for x in scene_ids_raw]

        meta_raw = f["metadata_json"][()]
        metadata_json = json.loads(_decode_str(meta_raw))

    if indices is not None:
        objects = objects[indices]
        scene_ids = [scene_ids[i] for i in indices]

    return {
        "objects": objects,
        "wavelengths_nm": wavelengths_nm,
        "scene_id": scene_ids,
        "metadata_json": metadata_json,
        "n_scenes": objects.shape[0],
        "n_channels": objects.shape[1],
        "spatial_size": objects.shape[-2:],
    }


def normalize_objects(objects: np.ndarray, per_scene: bool = True) -> np.ndarray:
    if per_scene:
        for i in range(objects.shape[0]):
            vmin = objects[i].min()
            vmax = objects[i].max()
            scale = vmax - vmin if vmax > vmin else 1.0
            objects[i] = (objects[i] - vmin) / scale
    else:
        vmin = objects.min()
        vmax = objects.max()
        scale = vmax - vmin if vmax > vmin else 1.0
        objects = (objects - vmin) / scale
    return objects.astype(np.float32)


def center_crop_objects(
    objects: np.ndarray,
    crop_size: Tuple[int, int],
) -> np.ndarray:
    _, _, h, w = objects.shape
    ch, cw = crop_size
    start_h = max(0, (h - ch) // 2)
    start_w = max(0, (w - cw) // 2)
    return objects[:, :, start_h:start_h + ch, start_w:start_w + cw]
