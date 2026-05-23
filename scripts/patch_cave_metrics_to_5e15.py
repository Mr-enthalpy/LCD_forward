from __future__ import annotations

import csv
import json
from pathlib import Path

HANDOFF = Path(r"D:\datasets\LCD_forward\lcd_forward_phase3_5_3_6_release_20260520")
SWEEP_JSON = HANDOFF / "thesis" / "alpha_sweep" / "cave_alpha_sweep.json"
CAVE_METRICS_DIR = HANDOFF / "thesis" / "phase3_6_linear_recon_cave" / "metrics"
TARGET_ALPHA_LABEL = "3e-15"
WAVELENGTHS = [450.0, 550.0, 650.0]

with open(SWEEP_JSON) as f:
    sweep = json.load(f)

scene_results = sweep["scene_results"]

summary = {"n_scenes": 0, "scenes": [], "all_cases_multi_greater_than_single": True, "alpha": 3e-15}
results_list = []

for scene_id in ["cd_ms", "clay_ms", "superballs_ms"]:
    entry = scene_results[scene_id][TARGET_ALPHA_LABEL]
    single = entry["single_frame_metrics"]
    multi = entry["multi_frame_metrics"]

    single_psnr = single["mean"]["psnr"]
    multi_psnr = multi["mean"]["psnr"]
    gain_db = round(multi_psnr - single_psnr, 2)

    summary["scenes"].append({
        "scene_id": scene_id,
        "single_psnr": single_psnr,
        "multi_psnr": multi_psnr,
        "gain_db": gain_db,
    })
    if gain_db <= 0:
        summary["all_cases_multi_greater_than_single"] = False

    results_list.append({
        "scene_id": scene_id,
        "single_frame_metrics": single,
        "multi_frame_metrics": multi,
    })

summary["n_scenes"] = len(summary["scenes"])
with open(CAVE_METRICS_DIR / "cave_recon_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Wrote cave_recon_summary.json ({summary['n_scenes']} scenes, alpha={TARGET_ALPHA_LABEL})")

with open(CAVE_METRICS_DIR / "cave_recon_metrics.json", "w") as f:
    json.dump({"n_scenes_evaluated": len(results_list), "results": results_list, "alpha": 3e-15}, f, indent=2)
print(f"Wrote cave_recon_metrics.json ({len(results_list)} scenes)")

def write_csv():
    rows = []
    for result in results_list:
        scene_id = result["scene_id"]
        single = result["single_frame_metrics"]
        multi = result["multi_frame_metrics"]
        methods = [("single_frame", single), ("multi_frame", multi)]

        single_mean_psnr = single["mean"]["psnr"]
        for method, metrics in methods:
            mean_gain = metrics["mean"]["psnr"] - single_mean_psnr if method == "multi_frame" else 0.0
            rows.append({
                "dataset": "cave",
                "scene_id": scene_id,
                "channel": "mean",
                "wavelength_nm": "",
                "method": method,
                "mse": metrics["mean"]["mse"],
                "relative_l2": metrics["mean"]["relative_l2"],
                "psnr": metrics["mean"]["psnr"],
                "correlation": metrics["mean"]["correlation"],
                "ssim_raw": metrics["mean"]["ssim_raw"],
                "ssim_display": metrics["mean"]["ssim_display"],
                "psnr_gain_vs_single_db": mean_gain,
            })
            for ch_str in ["0", "1", "2"]:
                ch = int(ch_str)
                ch_metrics = metrics["per_channel"][ch_str]
                single_ch_psnr = single["per_channel"][ch_str]["psnr"]
                gain = ch_metrics["psnr"] - single_ch_psnr if method == "multi_frame" else 0.0
                rows.append({
                    "dataset": "cave",
                    "scene_id": scene_id,
                    "channel": ch,
                    "wavelength_nm": WAVELENGTHS[ch] if ch < len(WAVELENGTHS) else "",
                    "method": method,
                    "mse": ch_metrics["mse"],
                    "relative_l2": ch_metrics["relative_l2"],
                    "psnr": ch_metrics["psnr"],
                    "correlation": ch_metrics["correlation"],
                    "ssim_raw": ch_metrics["ssim_raw"],
                    "ssim_display": ch_metrics["ssim_display"],
                    "psnr_gain_vs_single_db": gain,
                })

    fieldnames = [
        "dataset", "scene_id", "channel", "wavelength_nm", "method",
        "mse", "relative_l2", "psnr", "correlation",
        "ssim_raw", "ssim_display", "psnr_gain_vs_single_db",
    ]
    out_path = CAVE_METRICS_DIR / "reconstruction_metrics_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {out_path.name} ({len(rows)} rows)")

write_csv()

sweep["current_alpha"] = 3e-15
sweep["current_alpha_label"] = "3e-15"
with open(SWEEP_JSON, "w") as f:
    json.dump(sweep, f, indent=2)
print(f"Updated cave_alpha_sweep.json current_alpha -> 3e-15")
