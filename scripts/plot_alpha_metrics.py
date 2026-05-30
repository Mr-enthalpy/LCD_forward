from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SWEEP_JSON = Path("outputs/alpha_sweep/cave_alpha_sweep.json")
DEFAULT_OUTPUT_DIR = Path("outputs/alpha_sweep")

SCENE_COLORS = {
    "cd_ms": "#4c72b0",
    "clay_ms": "#55a868",
    "superballs_ms": "#c44e52",
}
SCENE_LABELS = {
    "cd_ms": "场景 cd",
    "clay_ms": "场景 clay",
    "superballs_ms": "场景 superballs",
}
MEAN_PSNR_COLOR = "#d62728"
MEAN_SSIM_RAW_COLOR = "#1f77b4"
MEAN_SSIM_DISPLAY_COLOR = "#9467bd"


def load_sweep_data(json_path: Path) -> dict[str, Any]:
    with json_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def extract_metric_curves(data: dict[str, Any]) -> dict[str, Any]:
    scene_results = data.get("scene_results", {})
    alphas = sorted(data.get("alphas", []))

    alpha_values = []
    mean_psnr, mean_ssim_raw, mean_ssim_display, mean_corr, mean_gain = [], [], [], [], []
    per_scene = {}

    for scene_id in scene_results:
        per_scene[scene_id] = {
            "alphas": [],
            "psnr": [],
            "ssim_raw": [],
            "ssim_display": [],
            "corr": [],
            "gain": [],
        }

    for alpha in alphas:
        psnrs, ssims_raw, ssims_disp, corrs, gains = [], [], [], [], []
        for scene_id, alpha_map in scene_results.items():
            alpha_label_key = None
            for key in alpha_map:
                if abs(alpha_map[key].get("alpha", 0) - alpha) < 1e-30:
                    alpha_label_key = key
                    break
            if alpha_label_key and alpha_label_key in alpha_map:
                entry = alpha_map[alpha_label_key]
                multi_m = entry.get("multi_frame_metrics", {})
                single_m = entry.get("single_frame_metrics", {})
                mean_mm = multi_m.get("mean", {})
                mean_sm = single_m.get("mean", {})
                psnr_v = mean_mm.get("psnr")
                if psnr_v is not None and math.isfinite(psnr_v):
                    psnrs.append(psnr_v)
                    ssims_raw.append(mean_mm.get("ssim_raw"))
                    ssims_disp.append(mean_mm.get("ssim_display"))
                    corrs.append(mean_mm.get("correlation"))
                    single_psnr = mean_sm.get("psnr", 0)
                    gains.append(psnr_v - single_psnr if single_psnr and math.isfinite(single_psnr) else None)
                    per_scene[scene_id]["alphas"].append(alpha)
                    per_scene[scene_id]["psnr"].append(psnr_v)
                    per_scene[scene_id]["ssim_raw"].append(mean_mm.get("ssim_raw"))
                    per_scene[scene_id]["ssim_display"].append(mean_mm.get("ssim_display"))
                    per_scene[scene_id]["corr"].append(mean_mm.get("correlation"))
                    per_scene[scene_id]["gain"].append(gains[-1])

        if psnrs:
            alpha_values.append(alpha)
            mean_psnr.append(float(np.mean(psnrs)))
            _ssim_raw_arr = np.array([v if v is not None else np.nan for v in ssims_raw], dtype=np.float64)
            _ssim_disp_arr = np.array([v if v is not None else np.nan for v in ssims_disp], dtype=np.float64)
            _corr_arr = np.array([v if v is not None else np.nan for v in corrs], dtype=np.float64)
            mean_ssim_raw.append(float(np.nanmean(_ssim_raw_arr)))
            mean_ssim_display.append(float(np.nanmean(_ssim_disp_arr)))
            mean_corr.append(float(np.nanmean(_corr_arr)))
            mean_gain.append(float(np.nanmean([g for g in gains if g is not None])) if any(g is not None for g in gains) else np.nan)
        else:
            alpha_values.append(alpha)
            mean_psnr.append(np.nan)
            mean_ssim_raw.append(np.nan)
            mean_ssim_display.append(np.nan)
            mean_corr.append(np.nan)
            mean_gain.append(np.nan)

    alpha_log10 = [math.log10(a) for a in alpha_values]
    return {
        "alphas": alpha_values,
        "alpha_log10": alpha_log10,
        "mean_psnr": mean_psnr,
        "mean_ssim_raw": mean_ssim_raw,
        "mean_ssim_display": mean_ssim_display,
        "mean_corr": mean_corr,
        "mean_gain": mean_gain,
        "per_scene": per_scene,
        "policy": data.get("policy", "adaptive"),
        "mask_ids": data.get("mask_ids", []),
    }


def _format_alpha_exponent(alpha: float) -> str:
    if alpha == 0.0:
        return "0"
    s = f"{alpha:.1e}"
    if "e" in s:
        mant, exp = s.split("e")
        mant = mant.rstrip("0").rstrip(".")
        return f"${mant}\\times10^{{{int(exp)}}}$"
    return s


def _compute_zone_boundaries(alphas, alpha_log10):
    valid_alpha_log10 = [l for a, l in zip(alphas, alpha_log10) if a >= 5e-16]
    collapse_right = math.log10(5e-16) + 0.15
    high_reg_left = math.log10(5e-7)
    if valid_alpha_log10:
        x_min = min(valid_alpha_log10) - 0.15
    else:
        x_min = math.log10(5e-16)
    x_max = max(alpha_log10) + 0.3
    return collapse_right, high_reg_left, x_min, x_max


def _add_zone_shading(ax, x_min, x_max, y_min, y_max, collapse_right, high_reg_left):
    ax.fill_betweenx(
        [y_min, y_max],
        x_min, collapse_right,
        color="gray", alpha=0.1,
    )
    ax.fill_betweenx(
        [y_min, y_max],
        collapse_right, high_reg_left,
        color="#2ca02c", alpha=0.06,
    )
    ax.fill_betweenx(
        [y_min, y_max],
        high_reg_left, x_max,
        color="#ff7f0e", alpha=0.08,
    )


def _add_zone_labels(ax, y_min, y_max, alpha_log10_range):
    y_label = y_max - (y_max - y_min) * 0.065
    cx = alpha_log10_range[0] + (math.log10(5e-16) - alpha_log10_range[0]) / 2
    ax.text(cx, y_label, "奇异值\n塌缩区",
            fontsize=7.5, ha="center", va="top", color="#555555",
            fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="gray", alpha=0.75))

    mid_cx = math.log10(5e-16) + (math.log10(5e-7) - math.log10(5e-16)) / 2
    ax.text(mid_cx, y_label, "稳定编码区",
            fontsize=7.5, ha="center", va="top", color="#1a7a1a",
            fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#2ca02c", alpha=0.75))


def _add_mid_frequency_label_below_legend(fig, ax, legend):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bbox = legend.get_window_extent(renderer=renderer).transformed(ax.transAxes.inverted())
    x = legend_bbox.x0
    y = max(0.02, legend_bbox.y0 - 0.035)
    ax.text(
        x,
        y,
        "中频编码\n被抑制",
        transform=ax.transAxes,
        fontsize=7.5,
        ha="left",
        va="top",
        color="#cc6600",
        fontstyle="italic",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#ff7f0e", alpha=0.75),
    )


def _add_best_alpha_vline(ax, best_alpha, best_val, y_min, y_max, color, annotate_x=None, annotate_y=None):
    x = math.log10(best_alpha)
    ax.axvline(x=x, color=color, linestyle="--", linewidth=1.6, alpha=0.85, zorder=4)
    if annotate_x is not None and annotate_y is not None:
        tx, ty = annotate_x, annotate_y
    else:
        tx, ty = x + 1.5, y_min + (y_max - y_min) * 0.12
    ax.annotate(
        f"最优 α={_format_alpha_exponent(best_alpha)}",
        xy=(x, best_val),
        xytext=(tx, ty),
        fontsize=7.5,
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0),
        color=color,
        fontweight="bold",
    )


def _best_alpha_annotation_xy(best_alpha, alpha_log10, y_min, y_max):
    x = math.log10(best_alpha)
    x_min = float(np.nanmin(alpha_log10))
    x_max = float(np.nanmax(alpha_log10))
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1e-6)
    if x > x_min + 0.65 * x_span:
        tx = x - 0.28 * x_span
    else:
        tx = x + 0.16 * x_span
    tx = min(max(tx, x_min + 0.06 * x_span), x_max - 0.06 * x_span)
    ty = y_max - 0.10 * y_span
    return tx, ty


def _setup_xaxis(ax, alpha_log10, xticks_log10):
    ax.set_xlim(alpha_log10.min() - 0.5, alpha_log10.max() + 0.5)
    xtk = sorted(set(xticks_log10))
    xtk = [x for x in xtk if alpha_log10.min() - 0.5 <= x <= alpha_log10.max() + 0.5]
    ax.set_xticks(xtk)
    ax.set_xticklabels([f"${int(x)}$" for x in xtk], fontsize=8.5)
    ax.set_xlabel(r"$\log_{10}(\alpha)$", fontsize=12)


def _add_per_scene_lines(ax, per_scene, metric_key, colors_map, alpha=0.35, lw=0.6, marker="o", ms=2.5, label_scenes=False):
    for scene_id, scene_data in per_scene.items():
        s_log10 = [math.log10(a) for a in scene_data["alphas"]]
        s_vals = scene_data[metric_key]
        if len(s_log10) < 2:
            continue
        color = colors_map.get(scene_id, "#888888")
        lbl = SCENE_LABELS.get(scene_id, scene_id) if label_scenes else None
        ax.plot(s_log10, s_vals, marker=marker, markersize=ms, linewidth=lw,
                color=color, alpha=alpha, label=lbl)


def plot_psnr(curves: dict[str, Any], output_dir: Path):
    alpha_log10 = np.array(curves["alpha_log10"])
    mean_psnr = np.array(curves["mean_psnr"])
    per_scene = curves["per_scene"]

    valid = ~np.isnan(mean_psnr)
    a_log10 = alpha_log10[valid]
    m_psnr = mean_psnr[valid]

    best_idx = int(np.nanargmax(m_psnr))
    best_alpha = 10.0 ** a_log10[best_idx]
    best_psnr = m_psnr[best_idx]

    all_data = [m_psnr] + [np.array(sd["psnr"]) for sd in per_scene.values() if sd["psnr"]]
    y_min = max(12, np.nanmin([np.nanmin(a) for a in all_data]) - 2)
    y_max = np.nanmax([np.nanmax(a) for a in all_data]) + 3

    collapse_right, high_reg_left, x_min, x_max = _compute_zone_boundaries(curves["alphas"], alpha_log10)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    _add_zone_shading(ax, x_min, x_max, y_min, y_max, collapse_right, high_reg_left)
    _add_zone_labels(ax, y_min, y_max, [alpha_log10.min(), alpha_log10.max()])
    _add_per_scene_lines(ax, per_scene, "psnr", SCENE_COLORS, alpha=0.18, lw=0.35, ms=1.5, label_scenes=True)
    annotate_x, annotate_y = _best_alpha_annotation_xy(best_alpha, alpha_log10, y_min, y_max)
    _add_best_alpha_vline(
        ax, best_alpha, best_psnr, y_min, y_max, MEAN_PSNR_COLOR,
        annotate_x=annotate_x, annotate_y=annotate_y,
    )

    ax.plot(a_log10, m_psnr, "s-", color=MEAN_PSNR_COLOR, linewidth=2.8,
            markersize=7, label="多帧平均峰值信噪比", zorder=6)

    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)

    ax.set_ylabel("峰值信噪比 (dB)", fontsize=13, fontweight="bold", color=MEAN_PSNR_COLOR)
    ax.tick_params(axis="y", labelcolor=MEAN_PSNR_COLOR, labelsize=10)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_ylim(y_min, y_max)
    _setup_xaxis(ax, alpha_log10, list(range(-18, 2, 2)) + [0])

    ax.set_title("峰值信噪比与正则化参数 α 的关系", fontsize=12, fontweight="bold", loc="left")

    fig.tight_layout(pad=1.2)
    _add_mid_frequency_label_below_legend(fig, ax, legend)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    for fmt in ("png", "pdf"):
        path = output_dir / f"alpha_psnr.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_ssim_raw(curves: dict[str, Any], output_dir: Path):
    alpha_log10 = np.array(curves["alpha_log10"])
    mean_ssim_raw = np.array(curves["mean_ssim_raw"])
    per_scene = curves["per_scene"]

    valid = ~np.isnan(mean_ssim_raw)
    if valid.sum() == 0:
        print("Skipping SSIM (raw) plot: no valid SSIM data in sweep")
        return
    a_log10 = alpha_log10[valid]
    m_raw = mean_ssim_raw[valid]

    best_idx = int(np.nanargmax(m_raw))
    best_alpha = 10.0 ** a_log10[best_idx]
    best_ssim_raw = m_raw[best_idx]

    all_data = [m_raw] + [np.array(sd["ssim_raw"]) for sd in per_scene.values() if sd["ssim_raw"]]
    y_min = max(0.0, np.nanmin([np.nanmin(a) for a in all_data]) - 0.04)
    y_max = min(1.02, np.nanmax([np.nanmax(a) for a in all_data]) + 0.02)

    collapse_right, high_reg_left, x_min, x_max = _compute_zone_boundaries(curves["alphas"], alpha_log10)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _add_zone_shading(ax, x_min, x_max, y_min, y_max, collapse_right, high_reg_left)
    _add_zone_labels(ax, y_min, y_max, [alpha_log10.min(), alpha_log10.max()])
    _add_per_scene_lines(ax, per_scene, "ssim_raw", SCENE_COLORS, alpha=0.18, lw=0.35, ms=1.5, label_scenes=True)
    annotate_x, annotate_y = _best_alpha_annotation_xy(best_alpha, alpha_log10, y_min, y_max)
    _add_best_alpha_vline(
        ax, best_alpha, best_ssim_raw, y_min, y_max, MEAN_SSIM_RAW_COLOR,
        annotate_x=annotate_x, annotate_y=annotate_y,
    )

    ax.plot(a_log10, m_raw, "D-", color=MEAN_SSIM_RAW_COLOR, linewidth=2.8,
            markersize=7, label="平均结构相似度（原始）", zorder=6)

    ax.set_ylabel("结构相似度", fontsize=13, fontweight="bold", color=MEAN_SSIM_RAW_COLOR)
    ax.tick_params(axis="y", labelcolor=MEAN_SSIM_RAW_COLOR, labelsize=10)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_ylim(y_min, y_max)
    _setup_xaxis(ax, alpha_log10, list(range(-18, 2, 2)) + [0])

    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)
    ax.set_title("原始结构相似度与正则化参数 α 的关系", fontsize=12, fontweight="bold", loc="left")

    fig.tight_layout(pad=1.2)
    _add_mid_frequency_label_below_legend(fig, ax, legend)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    for fmt in ("png", "pdf"):
        path = output_dir / f"alpha_ssim.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def plot_ssim_display(curves: dict[str, Any], output_dir: Path):
    alpha_log10 = np.array(curves["alpha_log10"])
    mean_ssim_display = np.array(curves["mean_ssim_display"])
    per_scene = curves["per_scene"]

    valid = ~np.isnan(mean_ssim_display)
    if valid.sum() == 0:
        print("Skipping SSIM (display) plot: no valid SSIM data in sweep")
        return
    a_log10 = alpha_log10[valid]
    m_disp = mean_ssim_display[valid]

    best_idx = int(np.nanargmax(m_disp))
    best_alpha = 10.0 ** a_log10[best_idx]
    best_ssim_disp = m_disp[best_idx]

    all_data = [m_disp] + [np.array(sd["ssim_display"]) for sd in per_scene.values() if sd["ssim_display"]]
    y_min = max(0.0, np.nanmin([np.nanmin(a) for a in all_data]) - 0.04)
    y_max = min(1.02, np.nanmax([np.nanmax(a) for a in all_data]) + 0.02)

    collapse_right, high_reg_left, x_min, x_max = _compute_zone_boundaries(curves["alphas"], alpha_log10)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    _add_zone_shading(ax, x_min, x_max, y_min, y_max, collapse_right, high_reg_left)
    _add_zone_labels(ax, y_min, y_max, [alpha_log10.min(), alpha_log10.max()])
    _add_per_scene_lines(ax, per_scene, "ssim_display", SCENE_COLORS, alpha=0.18, lw=0.35, ms=1.5, label_scenes=True)
    annotate_x, annotate_y = _best_alpha_annotation_xy(best_alpha, alpha_log10, y_min, y_max)
    _add_best_alpha_vline(
        ax, best_alpha, best_ssim_disp, y_min, y_max, MEAN_SSIM_DISPLAY_COLOR,
        annotate_x=annotate_x, annotate_y=annotate_y,
    )

    ax.plot(a_log10, m_disp, "^--", color=MEAN_SSIM_DISPLAY_COLOR, linewidth=2.8,
            markersize=7, label="平均结构相似度（显示）", zorder=6)

    ax.set_ylabel("结构相似度", fontsize=13, fontweight="bold", color=MEAN_SSIM_DISPLAY_COLOR)
    ax.tick_params(axis="y", labelcolor=MEAN_SSIM_DISPLAY_COLOR, labelsize=10)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.set_ylim(y_min, y_max)
    _setup_xaxis(ax, alpha_log10, list(range(-18, 2, 2)) + [0])

    legend = ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)
    ax.set_title("显示结构相似度与正则化参数 α 的关系", fontsize=12, fontweight="bold", loc="left")

    fig.tight_layout(pad=1.2)
    _add_mid_frequency_label_below_legend(fig, ax, legend)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    for fmt in ("png", "pdf"):
        path = output_dir / f"alpha_display_ssim.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"Saved {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot alpha-vs-metrics figures (PSNR, SSIM raw, display SSIM)")
    parser.add_argument("--sweep-json", type=Path, default=DEFAULT_SWEEP_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.sweep_json.exists():
        print(f"Sweep JSON not found: {args.sweep_json}")
        print("Run scripts/sweep_cave_recon_alpha.py first, then retry.")
        sys.exit(1)

    data = load_sweep_data(args.sweep_json)
    curves = extract_metric_curves(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_psnr(curves, args.output_dir)
    plot_ssim_raw(curves, args.output_dir)
    plot_ssim_display(curves, args.output_dir)


if __name__ == "__main__":
    main()
