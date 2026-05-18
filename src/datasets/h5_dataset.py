from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PathLike = Union[str, Path]


def _to_tensor(x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(x)).to(dtype=dtype)


class _BaseH5Dataset(Dataset):
    """
    Base HDF5 dataset wrapper.

    Expected dataset layout in one .h5 file:

        masks:  [N, T, 1, Hm, Wm]      float32
        psfs:   [N, T, L, Hp, Wp]      float32   (required for forward dataset)
        objects:[N, L, H, W]           float32   (required for recon dataset)
        frames: [N, T, 1, H, W]        float32   (optional for recon dataset)

    Optional metadata:
        wavelengths:        [L]
        spectral_response:  [L]

    Notes:
    - This class opens the file lazily inside __getitem__ to remain DataLoader-worker-safe.
    - It also supports optional subset indexing via `indices`.
    """

    def __init__(
        self,
        h5_path: PathLike,
        indices: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.h5_path = str(h5_path)
        self.dtype = dtype

        with h5py.File(self.h5_path, "r") as f:
            if "masks" not in f:
                raise KeyError(f"{self.h5_path} does not contain required key 'masks'")
            self._length = int(f["masks"].shape[0])
            self.available_keys = tuple(f.keys())

        if indices is None:
            self.indices = np.arange(self._length, dtype=np.int64)
        else:
            self.indices = np.asarray(indices, dtype=np.int64)
            if self.indices.ndim != 1:
                raise ValueError("`indices` must be a 1D sequence")
            if len(self.indices) == 0:
                raise ValueError("`indices` cannot be empty")
            if self.indices.min() < 0 or self.indices.max() >= self._length:
                raise IndexError(
                    f"indices out of range for dataset of length {self._length}"
                )

    def __len__(self) -> int:
        return int(len(self.indices))

    def _read_item(self, real_idx: int) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        with h5py.File(self.h5_path, "r") as f:
            for key in f.keys():
                ds = f[key]
                # sample-wise tensors have first dimension N
                if ds.ndim >= 1 and ds.shape[0] == self._length:
                    out[key] = _to_tensor(ds[real_idx], dtype=self.dtype)
                else:
                    # metadata tensors are returned as full tensors
                    out[key] = _to_tensor(ds[()], dtype=self.dtype)
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        real_idx = int(self.indices[idx])
        return self._read_item(real_idx)


class ForwardH5Dataset(_BaseH5Dataset):
    """
    Dataset for forward-model training/eval.

    Required keys:
        masks: [N, T, 1, Hm, Wm]
        psfs:  [N, T, L, Hp, Wp]

    Returned item:
        {
            "masks": Tensor[T, 1, Hm, Wm],
            "psfs": Tensor[T, L, Hp, Wp],
            ... optional metadata ...
        }
    """

    def __init__(
        self,
        h5_path: PathLike,
        indices: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(h5_path=h5_path, indices=indices, dtype=dtype)
        with h5py.File(self.h5_path, "r") as f:
            if "psfs" not in f:
                raise KeyError(
                    f"{self.h5_path} does not contain required key 'psfs' "
                    f"for ForwardH5Dataset"
                )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        return {
            "masks": item["masks"],
            "psfs": item["psfs"],
            **{k: v for k, v in item.items() if k not in {"masks", "psfs"}},
        }


class ReconH5Dataset(_BaseH5Dataset):
    """
    Dataset for reconstruction training/eval.

    Required keys:
        objects: [N, L, H, W]
        masks:   [N, T, 1, Hm, Wm]

    Optional:
        frames:  [N, T, 1, H, W]

    Returned item:
        {
            "objects": Tensor[L, H, W],
            "masks": Tensor[T, 1, Hm, Wm],
            "frames": Tensor[T, 1, H, W],   # if present
            ... optional metadata ...
        }
    """

    def __init__(
        self,
        h5_path: PathLike,
        indices: Optional[Sequence[int]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(h5_path=h5_path, indices=indices, dtype=dtype)
        with h5py.File(self.h5_path, "r") as f:
            if "objects" not in f:
                raise KeyError(
                    f"{self.h5_path} does not contain required key 'objects' "
                    f"for ReconH5Dataset"
                )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        required = {
            "objects": item["objects"],
            "masks": item["masks"],
        }
        if "frames" in item:
            required["frames"] = item["frames"]

        extras = {k: v for k, v in item.items() if k not in required}
        return {**required, **extras}