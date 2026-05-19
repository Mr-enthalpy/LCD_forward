from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _decode_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def verify_release_files(release_root: Path) -> list[str]:
    errors = []
    for name in ["RELEASE.json", "MANIFEST.json", "SHA256SUMS.txt"]:
        if not (release_root / name).exists():
            errors.append(f"Missing {name}")
    return errors


def verify_checksums(release_root: Path) -> list[str]:
    errors = []
    sums_path = release_root / "SHA256SUMS.txt"
    if not sums_path.exists():
        return ["SHA256SUMS.txt not found"]

    with open(sums_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            expected_hash, rel_path = parts[0], " ".join(parts[1:])
            target = release_root / rel_path
            if not target.exists():
                errors.append(f"Missing file for checksum: {rel_path}")
                continue
            sha = hashlib.sha256()
            with open(target, "rb") as fp:
                while True:
                    chunk = fp.read(8192)
                    if not chunk:
                        break
                    sha.update(chunk)
            actual = sha.hexdigest()
            if actual != expected_hash:
                errors.append(f"Checksum mismatch: {rel_path}")
    return errors


def verify_hdf5_schema(h5_path: Path, expected_n: int | None = None) -> list[str]:
    errors = []
    if not h5_path.exists():
        errors.append(f"Missing HDF5: {h5_path}")
        return errors

    try:
        with h5py.File(h5_path, "r") as f:
            required_keys = ["masks", "psfs", "wavelengths_nm", "mask_id", "mask_family", "metadata_json"]
            for key in required_keys:
                if key not in f:
                    errors.append(f"{h5_path.name}: missing key '{key}'")
                    continue

            if "masks" in f:
                masks = f["masks"]
                expected_shape = (None, 1, 1, 64, 64)
                actual = masks.shape
                if len(actual) < 2 or actual[-2:] != (64, 64):
                    errors.append(f"{h5_path.name}: masks shape {actual}, expected ...64x64")
                if str(masks.dtype) not in ("uint8", "|u1"):
                    errors.append(f"{h5_path.name}: masks dtype {masks.dtype}, expected uint8")
                if expected_n and actual[0] != expected_n:
                    errors.append(f"{h5_path.name}: N={actual[0]}, expected {expected_n}")

            if "psfs" in f:
                psfs = f["psfs"]
                if psfs.shape[-2:] != (512, 512):
                    errors.append(f"{h5_path.name}: psfs spatial {psfs.shape[-2:]}, expected 512x512")
                if psfs.shape[2] != 3:
                    errors.append(f"{h5_path.name}: psfs L={psfs.shape[2]}, expected 3")

            if "wavelengths_nm" in f:
                wl = np.asarray(f["wavelengths_nm"])
                expected = np.array([450.0, 550.0, 650.0])
                if not np.allclose(wl, expected, atol=0.1):
                    errors.append(f"{h5_path.name}: wavelengths_nm={wl.tolist()}, expected {expected.tolist()}")

            if "metadata_json" in f:
                meta_raw = _decode_str(f["metadata_json"][()])
                meta = json.loads(meta_raw)
                norm = meta.get("normalization", "unknown")
                roi = meta.get("psf_roi_key_used", "unknown")
                if "background_subtract_then_sum_normalize" not in norm:
                    errors.append(f"{h5_path.name}: normalization='{norm}', expected background_subtract_then_sum_normalize")
                if roi != "roi_512":
                    errors.append(f"{h5_path.name}: psf_roi_key_used='{roi}', expected roi_512")
    except Exception as e:
        errors.append(f"{h5_path.name}: error reading HDF5: {e}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify optic_system Phase 3 handoff")
    parser.add_argument("--release-root", required=True, help="Path to release root")
    parser.add_argument("--skip-checksum", action="store_true", help="Skip SHA-256 verification")
    parser.add_argument("--output-dir", default="outputs/bishe_first_pass", help="Output directory")
    args = parser.parse_args()

    release_root = Path(args.release_root)
    if not release_root.exists():
        print(f"ERROR: release root not found: {release_root}")
        sys.exit(1)

    all_errors = []

    all_errors.extend(verify_release_files(release_root))

    if not args.skip_checksum:
        all_errors.extend(verify_checksums(release_root))

    psf_dir = release_root / "lcd_forward" / "psf_dictionary"
    if not psf_dir.exists():
        all_errors.append(f"PSF dictionary dir not found: {psf_dir}")
    else:
        all_errors.extend(verify_hdf5_schema(psf_dir / "train.h5", expected_n=136))
        all_errors.extend(verify_hdf5_schema(psf_dir / "val.h5", expected_n=17))
        all_errors.extend(verify_hdf5_schema(psf_dir / "test.h5", expected_n=17))

    passed = len(all_errors) == 0

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "release_root": str(release_root),
        "passed": passed,
        "errors": all_errors,
        "checksum_skipped": args.skip_checksum,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = out_dir / "handoff_verification.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not passed:
        print(f"\nVerification FAILED with {len(all_errors)} error(s)")
        sys.exit(1)
    else:
        print(f"\nVerification PASSED")


if __name__ == "__main__":
    main()
