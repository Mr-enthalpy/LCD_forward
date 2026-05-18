from __future__ import annotations

from pathlib import Path
import sys

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    ax = np.arange(size, dtype=np.float32) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    k = k.astype(np.float32)
    k /= k.sum() + 1e-8
    return k


def make_sparse_object(rng: np.random.Generator, l: int, h: int, w: int) -> np.ndarray:
    obj = np.zeros((l, h, w), dtype=np.float32)
    for lam in range(l):
        num_points = int(rng.integers(8, 24))
        ys = rng.integers(0, h, size=num_points)
        xs = rng.integers(0, w, size=num_points)
        vals = rng.random(num_points, dtype=np.float32)
        obj[lam, ys, xs] = vals
    return obj


def make_periodic_mask(rng: np.random.Generator, hm: int, wm: int) -> np.ndarray:
    fy = int(rng.integers(2, 10))
    fx = int(rng.integers(2, 10))
    y = np.arange(hm, dtype=np.float32)[:, None]
    x = np.arange(wm, dtype=np.float32)[None, :]
    phase = rng.uniform(0, 2 * np.pi)
    pattern = 0.5 * (1.0 + np.sin(2 * np.pi * fy * y / hm + 2 * np.pi * fx * x / wm + phase))
    return pattern.astype(np.float32)


def make_sparse_mask(rng: np.random.Generator, hm: int, wm: int) -> np.ndarray:
    mask = np.zeros((hm, wm), dtype=np.float32)
    num_points = int(rng.integers(hm * wm // 40, hm * wm // 12))
    ys = rng.integers(0, hm, size=num_points)
    xs = rng.integers(0, wm, size=num_points)
    mask[ys, xs] = 1.0
    return mask


def make_random_mask(rng: np.random.Generator, hm: int, wm: int) -> np.ndarray:
    return rng.random((hm, wm), dtype=np.float32)


def mask_to_synthetic_psf(mask: np.ndarray, psf_size: int) -> np.ndarray:
    """
    Synthetic placeholder PSF generator for sample data.

    This is not intended to be physically faithful.
    It only provides a stable, normalized training-format target
    before real calibrated PSFs are available.
    """
    mean_val = float(mask.mean())
    std_val = float(mask.std())

    sigma = 1.0 + 3.0 * (1.0 - mean_val) + 1.5 * std_val
    sigma = max(sigma, 0.6)

    psf = gaussian_kernel(psf_size, sigma=sigma)

    # mild center modulation from low-order mask moments
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    mass = mask.sum() + 1e-8
    my = float((mask * yy).sum() / mass - cy)
    mx = float((mask * xx).sum() / mass - cx)

    shift_y = int(np.clip(round(my / max(mask.shape[0], 1) * psf_size * 0.5), -2, 2))
    shift_x = int(np.clip(round(mx / max(mask.shape[1], 1) * psf_size * 0.5), -2, 2))

    psf = np.roll(psf, shift=(shift_y, shift_x), axis=(0, 1))
    psf = psf / (psf.sum() + 1e-8)
    return psf.astype(np.float32)


def conv2d_same_numpy(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """
    Naive same-size spatial convolution for sample generation.
    x: [H, W]
    k: [Kh, Kw]
    """
    h, w = x.shape
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2

    xpad = np.pad(x, ((ph, ph), (pw, pw)), mode="constant")
    out = np.zeros_like(x, dtype=np.float32)

    # convolution kernel flip
    kf = np.flip(k, axis=(0, 1))

    for i in range(h):
        for j in range(w):
            patch = xpad[i:i + kh, j:j + kw]
            out[i, j] = np.sum(patch * kf, dtype=np.float32)

    return out


def render_frames_numpy(
    objects: np.ndarray,   # [L,H,W]
    psfs: np.ndarray,      # [T,L,Hp,Wp]
    spectral_response: np.ndarray | None = None,
) -> np.ndarray:
    t, l, _, _ = psfs.shape
    _, h, w = objects.shape

    if spectral_response is None:
        spectral_response = np.ones((l,), dtype=np.float32)

    frames = np.zeros((t, 1, h, w), dtype=np.float32)
    for ti in range(t):
        acc = np.zeros((h, w), dtype=np.float32)
        for li in range(l):
            acc += spectral_response[li] * conv2d_same_numpy(objects[li], psfs[ti, li])
        frames[ti, 0] = acc
    return frames


def generate_split(
    out_path: Path,
    n: int,
    t: int,
    l: int,
    hm: int,
    wm: int,
    hp: int,
    wp: int,
    h: int,
    w: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)

    masks = np.zeros((n, t, 1, hm, wm), dtype=np.float32)
    psfs = np.zeros((n, t, l, hp, wp), dtype=np.float32)
    objects = np.zeros((n, l, h, w), dtype=np.float32)
    frames = np.zeros((n, t, 1, h, w), dtype=np.float32)

    wavelengths = np.linspace(550.0, 550.0 + 10.0 * (l - 1), l, dtype=np.float32)
    spectral_response = np.ones((l,), dtype=np.float32)

    generators = [
        make_sparse_mask,
        make_periodic_mask,
        make_random_mask,
    ]

    for i in range(n):
        objects[i] = make_sparse_object(rng, l=l, h=h, w=w)

        for ti in range(t):
            gen = generators[(i + ti) % len(generators)]
            mask2d = gen(rng, hm, wm)
            masks[i, ti, 0] = mask2d

            for li in range(l):
                psf = mask_to_synthetic_psf(mask2d, psf_size=hp)
                psfs[i, ti, li] = psf

        frames[i] = render_frames_numpy(
            objects=objects[i],
            psfs=psfs[i],
            spectral_response=spectral_response,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("masks", data=masks)
        f.create_dataset("psfs", data=psfs)
        f.create_dataset("objects", data=objects)
        f.create_dataset("frames", data=frames)
        f.create_dataset("wavelengths", data=wavelengths)
        f.create_dataset("spectral_response", data=spectral_response)

    print(f"[prepare_sample_dataset] wrote: {out_path}")


def load_config() -> dict:
    cfg_path = ROOT / "configs" / "forward_single_lambda.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main() -> None:
    cfg = load_config()

    n_train = int(cfg.get("sample_n_train", 32))
    n_val = int(cfg.get("sample_n_val", 8))
    n_test = int(cfg.get("sample_n_test", 8))

    t = int(cfg.get("num_frames", 1))
    l = int(cfg.get("num_wavelengths", 1))

    hm = int(cfg.get("mask_size", 64))
    wm = int(cfg.get("mask_size", 64))
    hp = int(cfg.get("psf_size", 17))
    wp = int(cfg.get("psf_size", 17))
    h = int(cfg.get("image_size", 64))
    w = int(cfg.get("image_size", 64))

    base_seed = int(cfg.get("seed", 42))

    out_dir = ROOT / "data" / "sample"
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_split(
        out_path=out_dir / "train.h5",
        n=n_train,
        t=t,
        l=l,
        hm=hm,
        wm=wm,
        hp=hp,
        wp=wp,
        h=h,
        w=w,
        seed=base_seed,
    )
    generate_split(
        out_path=out_dir / "val.h5",
        n=n_val,
        t=t,
        l=l,
        hm=hm,
        wm=wm,
        hp=hp,
        wp=wp,
        h=h,
        w=w,
        seed=base_seed + 1,
    )
    generate_split(
        out_path=out_dir / "test.h5",
        n=n_test,
        t=t,
        l=l,
        hm=hm,
        wm=wm,
        hp=hp,
        wp=wp,
        h=h,
        w=w,
        seed=base_seed + 2,
    )

    print("[prepare_sample_dataset] done.")


if __name__ == "__main__":
    main()