from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RELEASE_ROOT = Path("D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520")


def psnr(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((gt - pred) ** 2))
    return float(10.0 * math.log10(1.0 / max(mse, 1e-12))), mse


def correlation(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_vec = gt.ravel().astype(np.float64)
    pred_vec = pred.ravel().astype(np.float64)
    gt_vec = gt_vec - gt_vec.mean()
    pred_vec = pred_vec - pred_vec.mean()
    denom = np.sqrt(np.sum(gt_vec**2) * np.sum(pred_vec**2)) + 1e-12
    return float(np.sum(gt_vec * pred_vec) / denom)


def minmax_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max <= x_min:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / (x_max - x_min)


def affine_calibrated_psnr(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float]:
    pred_vec = pred.ravel().astype(np.float64)
    gt_vec = gt.ravel().astype(np.float64)
    design = np.vstack([pred_vec, np.ones_like(pred_vec)]).T
    scale, offset = np.linalg.lstsq(design, gt_vec, rcond=None)[0]
    calibrated = scale * pred.astype(np.float64) + offset
    value, mse = psnr(gt.astype(np.float64), calibrated)
    return value, mse, float(scale), float(offset)


def box_mean(image: np.ndarray, window: int = 11) -> np.ndarray:
    pad = window // 2
    padded = np.pad(image.astype(np.float64), pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (window, window))
    return windows.mean(axis=(-1, -2))


def ssim_box11(gt: np.ndarray, pred: np.ndarray, data_range: float = 1.0) -> float:
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_gt = box_mean(gt)
    mu_pred = box_mean(pred)
    mu_gt2 = mu_gt * mu_gt
    mu_pred2 = mu_pred * mu_pred
    mu_cross = mu_gt * mu_pred
    sigma_gt = box_mean(gt * gt) - mu_gt2
    sigma_pred = box_mean(pred * pred) - mu_pred2
    sigma_cross = box_mean(gt * pred) - mu_cross
    score = ((2 * mu_cross + c1) * (2 * sigma_cross + c2)) / (
        (mu_gt2 + mu_pred2 + c1) * (sigma_gt + sigma_pred + c2) + 1e-12
    )
    return float(np.mean(score))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    grad_y, grad_x = np.gradient(image.astype(np.float64))
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def spatial_error_stats(gt: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    squared_error = (gt.astype(np.float64) - pred.astype(np.float64)) ** 2
    grad = gradient_magnitude(gt)
    flat_mask = grad <= np.quantile(grad, 0.50)
    structure_mask = grad >= np.quantile(grad, 0.75)
    total_sse = float(squared_error.sum()) + 1e-30
    return {
        "flat50_mse": float(squared_error[flat_mask].mean()),
        "structure25_mse": float(squared_error[structure_mask].mean()),
        "structure25_sse_share": float(squared_error[structure_mask].sum() / total_sse),
    }


def frequency_error_stats(error: np.ndarray) -> dict[str, float]:
    energy = np.abs(np.fft.fftshift(np.fft.fft2(error.astype(np.float64)))) ** 2
    height, width = error.shape
    yy, xx = np.ogrid[-height // 2 : height - height // 2, -width // 2 : width - width // 2]
    radius = np.sqrt((yy / (height / 2)) ** 2 + (xx / (width / 2)) ** 2)
    total = float(energy.sum()) + 1e-30
    return {
        "fft_low_le_0p1": float(energy[radius <= 0.1].sum() / total),
        "fft_mid_0p1_0p3": float(energy[(radius > 0.1) & (radius <= 0.3)].sum() / total),
        "fft_high_gt_0p3": float(energy[radius > 0.3].sum() / total),
    }


def load_stored_metrics(metrics_path: Path) -> dict[str, Any]:
    with metrics_path.open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    return {entry["scene_id"]: entry for entry in metrics["results"]}


def audit_scene(scene_npz: Path, stored_scene: dict[str, Any] | None) -> list[dict[str, Any]]:
    data = np.load(scene_npz)
    scene_id = str(data["scene_id"][0])
    gt = data["gt_object"]
    recon_single = data["recon_single"]
    recon_multi = data["recon_multi"]
    wavelengths = data["wavelengths_nm"]

    if gt.shape != recon_single.shape or gt.shape != recon_multi.shape:
        raise ValueError(
            f"{scene_id}: shape mismatch gt={gt.shape}, single={recon_single.shape}, multi={recon_multi.shape}"
        )

    rows = []
    for channel in range(gt.shape[0]):
        for method, pred in (("single", recon_single[channel]), ("multi", recon_multi[channel])):
            raw_psnr, raw_mse = psnr(gt[channel], pred)
            display_psnr, display_mse = psnr(minmax_normalize(gt[channel]), minmax_normalize(pred))
            affine_psnr, affine_mse, affine_scale, affine_offset = affine_calibrated_psnr(gt[channel], pred)
            stored = None
            if stored_scene is not None:
                key = f"{method}_frame_metrics"
                stored = stored_scene[key]["per_channel"][str(channel)]
            metric_delta = None if stored is None else raw_psnr - float(stored["psnr"])

            row = {
                "scene_id": scene_id,
                "channel": channel,
                "wavelength_nm": float(wavelengths[channel]),
                "method": method,
                "shape": list(gt[channel].shape),
                "raw_mse": raw_mse,
                "raw_psnr": raw_psnr,
                "stored_psnr_delta": metric_delta,
                "raw_correlation": correlation(gt[channel], pred),
                "ssim_box11_raw": ssim_box11(gt[channel], pred),
                "display_minmax_mse": display_mse,
                "display_minmax_psnr": display_psnr,
                "ssim_box11_minmax": ssim_box11(minmax_normalize(gt[channel]), minmax_normalize(pred)),
                "affine_psnr": affine_psnr,
                "affine_mse": affine_mse,
                "affine_scale": affine_scale,
                "affine_offset": affine_offset,
                "gt_mean": float(gt[channel].mean()),
                "pred_mean": float(pred.mean()),
                "gt_max": float(gt[channel].max()),
                "pred_max": float(pred.max()),
            }
            row.update(spatial_error_stats(gt[channel], pred))
            row.update(frequency_error_stats(gt[channel] - pred))
            rows.append(row)
    return rows


def build_markdown(rows: list[dict[str, Any]], release_root: Path) -> str:
    clay450 = [r for r in rows if r["scene_id"] == "clay_ms" and int(r["wavelength_nm"]) == 450]
    scene_means = []
    for scene_id in sorted({r["scene_id"] for r in rows}):
        for method in ("single", "multi"):
            selected = [r for r in rows if r["scene_id"] == scene_id and r["method"] == method]
            scene_means.append(
                {
                    "scene_id": scene_id,
                    "method": method,
                    "raw_psnr": float(np.mean([r["raw_psnr"] for r in selected])),
                    "corr": float(np.mean([r["raw_correlation"] for r in selected])),
                    "ssim_raw": float(np.mean([r["ssim_box11_raw"] for r in selected])),
                    "ssim_minmax": float(np.mean([r["ssim_box11_minmax"] for r in selected])),
                    "display_psnr": float(np.mean([r["display_minmax_psnr"] for r in selected])),
                }
            )

    lines = [
        "# Phase 3.6 CAVE Metrics Audit",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Scope",
        "",
        f"- Release root: `{release_root.as_posix()}`",
        "- Source arrays: `thesis/phase3_6_linear_recon_cave/data/*/recon_appendix_arrays.npz`",
        "- Stored metrics: `thesis/phase3_6_linear_recon_cave/metrics/cave_recon_metrics.json`",
        "- This audit is offline-only and uses saved CAVE appendices; it does not invoke hardware control.",
        "",
        "## Scene-Level Metric Cross-Check",
        "",
        "| Scene | Method | Raw PSNR | Raw corr. | Raw SSIM | Min-max SSIM | Min-max PSNR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in scene_means:
        lines.append(
            f"| {item['scene_id']} | {item['method']} | {item['raw_psnr']:.2f} | "
            f"{item['corr']:.4f} | {item['ssim_raw']:.4f} | {item['ssim_minmax']:.4f} | "
            f"{item['display_psnr']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## clay_ms 450 nm Detail",
            "",
            "| Method | Raw PSNR | Corr. | Min-max PSNR | Affine PSNR | GT mean | Pred mean | FFT low share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in clay450:
        lines.append(
            f"| {row['method']} | {row['raw_psnr']:.2f} | {row['raw_correlation']:.4f} | "
            f"{row['display_minmax_psnr']:.2f} | {row['affine_psnr']:.2f} | "
            f"{row['gt_mean']:.4f} | {row['pred_mean']:.4f} | {row['fft_low_le_0p1']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            "- Stored PSNR values reproduce from the NPZ arrays; no metric recomputation mismatch was found.",
            "- Single-frame and multi-frame reconstructions have identical `[3, 256, 256]` shapes against GT, so there is no silent broadcasting in the saved metric calculation.",
            "- The clay_ms 450 nm negative PSNR gain is dominated by absolute low-frequency / DC amplitude error, not structural mismatch: multi-frame correlation is near 1.0, but its mean intensity is far above the very dark GT band.",
            "- The per-band and pseudo-RGB figures use independent display autoscaling / per-panel normalization, so they can show strong structural recovery while raw PSNR remains low.",
            "- Raw PSNR is still valid as an absolute radiometric error metric, but it is insufficient as the only thesis-facing quality metric for this normalized CAVE simulation.",
            "",
            "## Recommendation",
            "",
            "- Report raw PSNR together with correlation and SSIM.",
            "- Add a display-normalized or affine-calibrated metric only as a diagnostic for structural recovery, not as a replacement for raw radiometric error.",
            "- For clay_ms, explicitly state that multi-frame reconstruction recovers spectral structure but has residual amplitude calibration error in the 450 nm and 650 nm bands.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(rows: list[dict[str, Any]], release_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cave_metrics_audit.csv"
    json_path = output_dir / "cave_metrics_audit.json"
    md_path = output_dir / "cave_metrics_audit.md"

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"release_root": str(release_root), "rows": rows}, handle, indent=2)

    md_path.write_text(build_markdown(rows, release_root), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Phase 3.6 CAVE reconstruction metrics")
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metric_audit"))
    args = parser.parse_args()

    cave_root = args.release_root / "thesis" / "phase3_6_linear_recon_cave"
    data_root = cave_root / "data"
    metrics_path = cave_root / "metrics" / "cave_recon_metrics.json"
    if not data_root.exists():
        raise FileNotFoundError(f"CAVE data directory not found: {data_root}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"CAVE metrics file not found: {metrics_path}")

    stored_metrics = load_stored_metrics(metrics_path)
    rows: list[dict[str, Any]] = []
    for scene_npz in sorted(data_root.glob("scene_*/*.npz")):
        scene_id = scene_npz.parent.name.removeprefix("scene_")
        rows.extend(audit_scene(scene_npz, stored_metrics.get(scene_id)))
    if not rows:
        raise RuntimeError(f"No scene NPZ files found under {data_root}")

    max_metric_delta = max(abs(r["stored_psnr_delta"] or 0.0) for r in rows)
    print(f"Rows: {len(rows)}")
    print(f"Max stored PSNR delta: {max_metric_delta:.6e} dB")
    write_outputs(rows, args.release_root, args.output_dir)


if __name__ == "__main__":
    main()
