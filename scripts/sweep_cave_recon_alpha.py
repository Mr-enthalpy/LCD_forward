from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.optic_handoff import load_psf_dictionary
from src.utils.metrics import ssim_box11, minmax_normalize


DEFAULT_RELEASE_ROOT = Path("D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520")
DEFAULT_TRAIN_H5 = Path("D:/datasets/optic_system/phase3_release_20260520/lcd_forward/psf_dictionary/train.h5")
DEFAULT_ALPHAS = [
    1e-18,
    3e-16,
    5e-16,
    7e-16,
    1e-15,
    1.5e-15,
    2e-15,
    2.5e-15,
    3e-15,
    4e-15,
    5e-15,
    7e-15,
    1e-14,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1e+0,
]


def _decode_array_strings(values: np.ndarray) -> list[str]:
    out = []
    for value in values:
        if isinstance(value, bytes):
            out.append(value.decode("utf-8"))
        else:
            out.append(str(value))
    return out


def select_psfs_by_mask_ids(train_h5: Path, mask_ids: list[str], psf_size: tuple[int, int]) -> torch.Tensor:
    data = load_psf_dictionary(train_h5, psf_working_size=psf_size)
    index = {mask_id: idx for idx, mask_id in enumerate(data["mask_id"])}
    missing = [mask_id for mask_id in mask_ids if mask_id not in index]
    if missing:
        raise KeyError(f"Selected mask IDs not found in PSF dictionary: {missing}")
    psfs = data["psfs"][[index[mask_id] for mask_id in mask_ids], 0]
    return torch.from_numpy(psfs).float()


def batched_frequency_ridge(
    frames: torch.Tensor,
    psfs: torch.Tensor,
    alpha: float,
    policy: str,
    cond_threshold: float = 1e3,
) -> torch.Tensor:
    if frames.ndim != 3:
        raise ValueError(f"frames must be [T,H,W], got {frames.shape}")
    if psfs.ndim != 4:
        raise ValueError(f"psfs must be [T,L,H,W], got {psfs.shape}")

    n_frames, height, width = frames.shape
    n_psf_frames, n_lambda, psf_height, psf_width = psfs.shape
    if (n_frames, height, width) != (n_psf_frames, psf_height, psf_width):
        raise ValueError(f"shape mismatch frames={frames.shape}, psfs={psfs.shape}")

    frames_fft = torch.fft.fft2(frames.double()).reshape(n_frames, -1).T.to(torch.complex128)
    psfs_fft = torch.fft.fft2(psfs.double()).reshape(n_frames, n_lambda, -1).permute(2, 0, 1).to(torch.complex128)
    h_conj = psfs_fft.conj().transpose(1, 2)
    hth = h_conj @ psfs_fft
    hty = h_conj @ frames_fft.unsqueeze(-1)
    eye = torch.eye(n_lambda, dtype=torch.complex128, device=frames.device).unsqueeze(0)
    h_norm = hth.abs().amax(dim=(1, 2)).real + 1e-12

    if policy == "none" or alpha == 0.0:
        reg_scale = torch.full_like(h_norm, 1e-12) * h_norm
    elif policy == "threshold":
        singular_values = torch.linalg.svdvals(hth)
        cond = (singular_values[:, 0] / (singular_values[:, -1] + 1e-12)).real
        reg_scale = torch.where(cond < cond_threshold, torch.zeros_like(h_norm), alpha * h_norm)
    elif policy == "adaptive":
        reg_scale = alpha * h_norm
    else:
        raise ValueError(f"unknown policy: {policy}")

    solution = torch.linalg.solve(hth + reg_scale[:, None, None] * eye, hty).squeeze(-1)
    recon_fft = solution.T.reshape(n_lambda, height, width)
    recon = torch.fft.ifft2(recon_fft).real
    return torch.clamp(recon, 0.0, None).float()


def compute_metrics(gt: torch.Tensor, pred: torch.Tensor) -> dict[str, Any]:
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    rows = {}
    psnrs, mses, corrs, ssims_raw, ssims_display, rel_l2s = [], [], [], [], [], []
    for channel in range(gt_np.shape[0]):
        gt_ch = gt_np[channel]
        pred_ch = pred_np[channel]
        mse = float(np.mean((gt_ch - pred_ch) ** 2))
        psnr_val = float(10.0 * math.log10(1.0 / max(mse, 1e-12)))
        rel_l2 = float(
            np.linalg.norm(gt_ch - pred_ch) / (np.linalg.norm(gt_ch) + 1e-8)
        )
        gt_flat = gt_ch.ravel()
        pred_flat = pred_ch.ravel()
        gt_centered = gt_flat - gt_flat.mean()
        pred_centered = pred_flat - pred_flat.mean()
        corr = float(
            np.sum(gt_centered * pred_centered)
            / (np.sqrt(np.sum(gt_centered ** 2) * np.sum(pred_centered ** 2)) + 1e-8)
        )
        ssim_raw = ssim_box11(gt_ch, pred_ch)
        gt_display = minmax_normalize(gt_ch)
        pred_display = minmax_normalize(pred_ch)
        ssim_disp = ssim_box11(gt_display, pred_display)

        rows[str(channel)] = {
            "mse": mse,
            "psnr": psnr_val,
            "relative_l2": rel_l2,
            "correlation": corr,
            "ssim_raw": ssim_raw,
            "ssim_display": ssim_disp,
        }
        mses.append(mse)
        psnrs.append(psnr_val)
        rel_l2s.append(rel_l2)
        corrs.append(corr)
        ssims_raw.append(ssim_raw)
        ssims_display.append(ssim_disp)
    return {
        "per_channel": rows,
        "mean": {
            "mse": float(np.mean(mses)),
            "psnr": float(np.mean(psnrs)),
            "relative_l2": float(np.mean(rel_l2s)),
            "correlation": float(np.mean(corrs)),
            "ssim_raw": float(np.mean(ssims_raw)),
            "ssim_display": float(np.mean(ssims_display)),
        },
    }


def format_alpha(alpha: float) -> str:
    if alpha == 0.0:
        return "0"
    mantissa, exponent = f"{alpha:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def load_scene_npzs(release_root: Path) -> list[Path]:
    data_root = release_root / "thesis" / "phase3_6_linear_recon_cave" / "data"
    scene_npzs = sorted(data_root.glob("scene_*/*.npz"))
    if not scene_npzs:
        data_root = release_root / "linear_recon_cave"
        scene_npzs = sorted(data_root.glob("scene_*/*.npz"))
    if not scene_npzs:
        raise FileNotFoundError(f"No CAVE appendix arrays found under {data_root}")
    return scene_npzs


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    scene_npzs = load_scene_npzs(args.release_root)
    first = np.load(scene_npzs[0])
    mask_ids = _decode_array_strings(first["selected_mask_ids"])
    gt_shape = first["gt_object"].shape
    psf_size = tuple(gt_shape[-2:])
    psfs = select_psfs_by_mask_ids(args.train_h5, mask_ids, psf_size).to(args.device)

    rows = []
    scene_results = {}
    for scene_npz in scene_npzs:
        data = np.load(scene_npz)
        scene_id = str(data["scene_id"][0])
        gt = torch.from_numpy(data["gt_object"]).float().to(args.device)
        frames = torch.from_numpy(data["rendered_frames"]).float().to(args.device)
        scene_results[scene_id] = {}
        print(f"Scene {scene_id}: {gt.shape}, frames={frames.shape}")

        for alpha in args.alphas:
            try:
                recon_multi = batched_frequency_ridge(frames, psfs, alpha=alpha, policy=args.policy)
                recon_single = batched_frequency_ridge(frames[:1], psfs[:1], alpha=alpha, policy=args.policy)
                multi_metrics = compute_metrics(gt, recon_multi)
                single_metrics = compute_metrics(gt, recon_single)
                scene_results[scene_id][format_alpha(alpha)] = {
                    "alpha": alpha,
                    "policy": args.policy,
                    "single_frame_metrics": single_metrics,
                    "multi_frame_metrics": multi_metrics,
                }
                rows.append(
                    {
                        "scene_id": scene_id,
                        "alpha": alpha,
                        "alpha_label": format_alpha(alpha),
                        "policy": args.policy,
                        "status": "ok",
                        "error": "",
                        "single_psnr": single_metrics["mean"]["psnr"],
                        "multi_psnr": multi_metrics["mean"]["psnr"],
                        "gain_psnr": multi_metrics["mean"]["psnr"] - single_metrics["mean"]["psnr"],
                        "single_corr": single_metrics["mean"]["correlation"],
                        "multi_corr": multi_metrics["mean"]["correlation"],
                        "single_mse": single_metrics["mean"]["mse"],
                        "multi_mse": multi_metrics["mean"]["mse"],
                        "single_ssim_raw": single_metrics["mean"]["ssim_raw"],
                        "multi_ssim_raw": multi_metrics["mean"]["ssim_raw"],
                        "single_ssim_display": single_metrics["mean"]["ssim_display"],
                        "multi_ssim_display": multi_metrics["mean"]["ssim_display"],
                    }
                )
                print(
                    f"  alpha={format_alpha(alpha):>5s} "
                    f"single={single_metrics['mean']['psnr']:.2f} "
                    f"multi={multi_metrics['mean']['psnr']:.2f} "
                    f"gain={multi_metrics['mean']['psnr'] - single_metrics['mean']['psnr']:+.2f} "
                    f"corr={multi_metrics['mean']['correlation']:.4f} "
                    f"ssim={multi_metrics['mean']['ssim_raw']:.4f}"
                )
            except RuntimeError as exc:
                rows.append(
                    {
                        "scene_id": scene_id,
                        "alpha": alpha,
                        "alpha_label": format_alpha(alpha),
                        "policy": args.policy,
                        "status": "failed",
                        "error": str(exc),
                        "single_psnr": np.nan,
                        "multi_psnr": np.nan,
                        "gain_psnr": np.nan,
                        "single_corr": np.nan,
                        "multi_corr": np.nan,
                        "single_mse": np.nan,
                        "multi_mse": np.nan,
                        "single_ssim_raw": np.nan,
                        "multi_ssim_raw": np.nan,
                        "single_ssim_display": np.nan,
                        "multi_ssim_display": np.nan,
                    }
                )
                print(f"  alpha={format_alpha(alpha):>5s} FAILED: {exc}")

    summary = []
    for alpha in args.alphas:
        selected = [row for row in rows if row["alpha"] == alpha and row["status"] == "ok"]
        failed = [row for row in rows if row["alpha"] == alpha and row["status"] != "ok"]
        if failed:
            summary.append(
                {
                    "alpha": alpha,
                    "alpha_label": format_alpha(alpha),
                    "policy": args.policy,
                    "n_scenes_ok": len(selected),
                    "n_scenes_failed": len(failed),
                    "mean_single_psnr": np.nan,
                    "mean_multi_psnr": np.nan,
                    "mean_gain_psnr": np.nan,
                    "mean_single_corr": np.nan,
                    "mean_multi_corr": np.nan,
                    "mean_multi_mse": np.nan,
                    "mean_multi_ssim_raw": np.nan,
                    "mean_multi_ssim_display": np.nan,
                }
            )
            continue
        summary.append(
            {
                "alpha": alpha,
                "alpha_label": format_alpha(alpha),
                "policy": args.policy,
                "n_scenes_ok": len(selected),
                "n_scenes_failed": 0,
                "mean_single_psnr": float(np.mean([row["single_psnr"] for row in selected])),
                "mean_multi_psnr": float(np.mean([row["multi_psnr"] for row in selected])),
                "mean_gain_psnr": float(np.mean([row["gain_psnr"] for row in selected])),
                "mean_single_corr": float(np.mean([row["single_corr"] for row in selected])),
                "mean_multi_corr": float(np.mean([row["multi_corr"] for row in selected])),
                "mean_multi_mse": float(np.mean([row["multi_mse"] for row in selected])),
                "mean_multi_ssim_raw": float(np.mean([row["multi_ssim_raw"] for row in selected])),
                "mean_multi_ssim_display": float(np.mean([row["multi_ssim_display"] for row in selected])),
            }
        )

    valid_summary = [row for row in summary if row["n_scenes_failed"] == 0]
    best_by_multi_psnr = max(valid_summary, key=lambda row: row["mean_multi_psnr"])
    best_by_gain = max(valid_summary, key=lambda row: row["mean_gain_psnr"])
    return {
        "release_root": str(args.release_root),
        "train_h5": str(args.train_h5),
        "policy": args.policy,
        "alphas": args.alphas,
        "mask_ids": mask_ids,
        "scene_results": scene_results,
        "rows": rows,
        "summary": summary,
        "best_by_mean_multi_psnr": best_by_multi_psnr,
        "best_by_mean_gain_psnr": best_by_gain,
        "current_alpha": args.current_alpha,
        "current_alpha_label": format_alpha(args.current_alpha),
    }


def write_outputs(result: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    summary = result["summary"]
    with (output_dir / "cave_alpha_sweep_by_scene.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "cave_alpha_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with (output_dir / "cave_alpha_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    lines = [
        "# CAVE Alpha Sweep",
        "",
        f"- Policy: `{result['policy']}`",
        f"- Current alpha: `{result['current_alpha_label']}`",
        f"- Best by mean multi-frame raw PSNR: `{result['best_by_mean_multi_psnr']['alpha_label']}`",
        f"- Best by mean PSNR gain: `{result['best_by_mean_gain_psnr']['alpha_label']}`",
        "",
        "| Alpha | Mean multi PSNR | Mean gain | Mean multi corr | Mean multi SSIM raw | Mean multi SSIM display |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        if row["n_scenes_failed"]:
            lines.append(f"| {row['alpha_label']} | failed | failed | failed | failed | failed |")
        else:
            ssim_raw = row.get("mean_multi_ssim_raw", np.nan)
            ssim_disp = row.get("mean_multi_ssim_display", np.nan)
            ssim_raw_str = f"{ssim_raw:.4f}" if not np.isnan(ssim_raw) else "N/A"
            ssim_disp_str = f"{ssim_disp:.4f}" if not np.isnan(ssim_disp) else "N/A"
            lines.append(
                f"| {row['alpha_label']} | {row['mean_multi_psnr']:.2f} | "
                f"{row['mean_gain_psnr']:+.2f} | {row['mean_multi_corr']:.4f} | "
                f"{ssim_raw_str} | {ssim_disp_str} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The primary selection metric is mean multi-frame raw PSNR across the three saved CAVE scenes.",
            "SSIM (raw) measures structural similarity on the original value range.",
            "SSIM (display) measures structural similarity after per-channel min-max normalization —",
            "this separates structural recovery from amplitude/DC offset errors.",
            "Correlation is reported as a structural companion metric.",
            "",
            "The optimal alpha is the smallest value that numerically stabilizes the per-frequency",
            "linear solves (avoiding singular-value collapse at DC) without meaningfully attenuating",
            "the mid-frequency encoding that carries channel-separation information.",
            "See docs/alpha_interpretability.md for the full physical explanation.",
        ]
    )
    (output_dir / "cave_alpha_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep Phase 3.6 CAVE reconstruction alpha values")
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--train-h5", type=Path, default=DEFAULT_TRAIN_H5)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/alpha_sweep"))
    parser.add_argument("--policy", default="adaptive", choices=["adaptive", "threshold", "none"])
    parser.add_argument("--alphas", nargs="*", type=float, default=DEFAULT_ALPHAS)
    parser.add_argument("--current-alpha", type=float, default=5e-15)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_sweep(args)
    write_outputs(result, args.output_dir)
    print("Best by mean multi-frame raw PSNR:", result["best_by_mean_multi_psnr"])
    print("Best by mean PSNR gain:", result["best_by_mean_gain_psnr"])


if __name__ == "__main__":
    main()
