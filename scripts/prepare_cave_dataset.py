from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CAVE_WAVELENGTHS_NM = np.arange(400, 701, 10, dtype=np.float32)


def find_wavelength_indices(
    target_wl: np.ndarray,
    source_wl: np.ndarray,
    method: str = "nearest",
) -> dict:
    indices = []
    weights = []
    for tw in target_wl:
        diffs = np.abs(source_wl - tw)
        if method == "nearest":
            idx = int(np.argmin(diffs))
            indices.append(idx)
            weights.append(1.0)
        elif method == "linear":
            lo = int(np.floor(np.searchsorted(source_wl, tw)))
            lo = max(0, min(lo, len(source_wl) - 1))
            hi = min(lo + 1, len(source_wl) - 1)
            if lo == hi:
                indices.append(lo)
                weights.append(1.0)
            else:
                t = (tw - source_wl[lo]) / (source_wl[hi] - source_wl[lo] + 1e-12)
                indices.append(lo)
                weights.append(1.0 - t if t < 0.5 else t)
    return {
        "selection_method": method,
        "source_wavelengths_nm": source_wl.tolist(),
        "target_wavelengths_nm": target_wl.tolist(),
        "selected_indices": indices,
        "weights": weights,
    }


def load_scene_pngs(scene_dir: Path) -> np.ndarray | None:
    pngs = sorted(scene_dir.glob("*_ms_*.png"))
    if not pngs:
        pngs = sorted(scene_dir.glob("*.png"))
        pngs = [p for p in pngs if "_RGB" not in p.name and "Thumbs" not in p.name]
    if len(pngs) < 31:
        return None

    pngs = [p for p in pngs if "_ms_" in p.name or p.name.split(".")[0].endswith(tuple(f"_{i:02d}" for i in range(1, 32)))]
    if len(pngs) < 31:
        return None
    pngs = pngs[:31]
    bands = []
    for p in pngs:
        img = Image.open(p)
        if img.mode in ("RGB", "RGBA", "L", "P"):
            img = img.convert("L")
        img = np.array(img).astype(np.float32)
        bands.append(img)
    return np.stack(bands, axis=0)


def extract_three_channels(
    cube: np.ndarray,
    wl_info: dict,
) -> np.ndarray:
    indices = wl_info["selected_indices"]
    weights = wl_info["weights"]
    out = np.zeros((3, *cube.shape[-2:]), dtype=np.float32)
    for ch in range(3):
        idx = indices[ch]
        if ch < len(weights) and weights[ch] < 1.0 and idx + 1 < cube.shape[0]:
            out[ch] = (1 - weights[ch]) * cube[idx] + weights[ch] * cube[idx + 1]
        else:
            out[ch] = cube[idx]
    out = np.maximum(out, 0.0)
    return out


def center_crop(cube: np.ndarray, size: tuple) -> np.ndarray:
    _, h, w = cube.shape
    ch, cw = size
    sh = max(0, (h - ch) // 2)
    sw = max(0, (w - cw) // 2)
    return cube[:, sh:sh + ch, sw:sw + cw]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CAVE dataset for LCD_forward")
    parser.add_argument("--raw-dir", required=True, help="CAVE raw directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed HDF5")
    parser.add_argument("--target-wavelengths", nargs=3, type=float,
                        default=[450.0, 550.0, 650.0])
    parser.add_argument("--crop-size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_wl = np.array(args.target_wavelengths, dtype=np.float32)

    wl_info = find_wavelength_indices(target_wl, CAVE_WAVELENGTHS_NM, method="nearest")

    scene_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    print(f"Found {len(scene_dirs)} scene directories")

    objects_list = []
    scene_ids = []

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        inner_dir = scene_dir / scene_name
        if inner_dir.is_dir():
            png_dir = inner_dir
        else:
            png_dir = scene_dir

        cube = load_scene_pngs(png_dir)
        if cube is None:
            print(f"  SKIP {scene_name}: insufficient PNGs ({len(list(png_dir.glob('*.png')))})")
            continue

        three_ch = extract_three_channels(cube, wl_info)
        three_ch = center_crop(three_ch, tuple(args.crop_size))

        vmin, vmax = three_ch.min(), three_ch.max()
        if vmax > vmin:
            three_ch = (three_ch - vmin) / (vmax - vmin)
        else:
            three_ch = np.zeros_like(three_ch)

        objects_list.append(three_ch)
        scene_ids.append(scene_name)
        print(f"  OK {scene_name}: shape={three_ch.shape}")

    if not objects_list:
        print("ERROR: no valid scenes found")
        sys.exit(1)

    objects = np.stack(objects_list, axis=0).astype(np.float32)
    n_total = objects.shape[0]

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_total)

    n_train = max(1, int(n_total * args.train_split))
    n_val = max(1, int(n_total * args.val_split))
    n_test = n_total - n_train - n_val

    splits = {
        "train": perm[:n_train].tolist(),
        "val": perm[n_train:n_train + n_val].tolist(),
        "test": perm[n_train + n_val:].tolist(),
    }

    metadata = {
        "source": "CAVE Multispectral Image Database",
        "source_url": "https://cave.cs.columbia.edu/repository/Multispectral",
        **wl_info,
        "crop_size": args.crop_size,
        "normalization": "per_scene_minmax",
        "n_scenes": n_total,
        "split_counts": {"train": n_train, "val": n_val, "test": n_test},
    }

    for split_name, split_indices in splits.items():
        split_objects = objects[split_indices]
        split_ids = [scene_ids[i] for i in split_indices]

        out_path = out_dir / f"{split_name}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("objects", data=split_objects)
            f.create_dataset("wavelengths_nm", data=target_wl)
            f.create_dataset("scene_id", data=np.array(split_ids, dtype=h5py.string_dtype()))
            f.create_dataset("metadata_json", data=json.dumps(metadata))

        print(f"  Wrote {out_path}: {split_objects.shape}")

    index_path = out_dir / "cave_31band_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"splits": splits, **metadata}, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {index_path}")

    print(f"\nDone. Train={n_train}, Val={n_val}, Test={n_test}")


if __name__ == "__main__":
    main()
