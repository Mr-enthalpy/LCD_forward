from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REQUIRED_STRUCTURE = {
    "RELEASE.json": "file",
    "MANIFEST.json": "file",
    "SHA256SUMS.txt": "file",
    "data_contract.md": "file",
    "README.md": "file",
    "provenance/optic_system_release_reference.json": "file",
    "provenance/lcd_forward_run_manifest.json": "file",
    "provenance/bishe_first_pass.yaml": "file",
    "provenance/phase3_6_debug_and_tuning.md": "file",
    "thesis/phase3_5_forward_validation/figures/measured_vs_predicted_examples.png": "file",
    "thesis/phase3_5_forward_validation/metrics/psf_prediction_metrics.json": "file",
    "thesis/phase3_5_forward_validation/reports/forward_validation_report.md": "file",
    "thesis/h_matrix_diagnostics/figures/h_rank_map.png": "file",
    "thesis/h_matrix_diagnostics/figures/h_log_condition_map.png": "file",
    "thesis/h_matrix_diagnostics/figures/h_condition_histogram.png": "file",
    "thesis/h_matrix_diagnostics/metrics/h_matrix_diagnostics.json": "file",
    "thesis/h_matrix_diagnostics/reports/h_matrix_diagnostics_report.md": "file",
    "thesis/phase3_6_linear_recon_synthetic/figures/recon_comparison.png": "file",
    "thesis/phase3_6_linear_recon_synthetic/figures/recon_per_band_comparison.png": "file",
    "thesis/phase3_6_linear_recon_synthetic/figures/recon_rgb_pseudocolor_comparison.png": "file",
    "thesis/phase3_6_linear_recon_synthetic/data/recon_appendix_arrays.npz": "file",
    "thesis/phase3_6_linear_recon_synthetic/metrics/reconstruction_metrics.json": "file",
    "thesis/phase3_6_linear_recon_synthetic/reports/linear_recon_report.md": "file",
    "thesis/phase3_6_linear_recon_cave/figures/scene_cd_ms/recon_per_band_comparison.png": "file",
    "thesis/phase3_6_linear_recon_cave/figures/scene_cd_ms/recon_rgb_pseudocolor_comparison.png": "file",
    "thesis/phase3_6_linear_recon_cave/data/scene_cd_ms/recon_appendix_arrays.npz": "file",
    "thesis/phase3_6_linear_recon_cave/figures/scene_clay_ms/recon_per_band_comparison.png": "file",
    "thesis/phase3_6_linear_recon_cave/figures/scene_clay_ms/recon_rgb_pseudocolor_comparison.png": "file",
    "thesis/phase3_6_linear_recon_cave/data/scene_clay_ms/recon_appendix_arrays.npz": "file",
    "thesis/phase3_6_linear_recon_cave/figures/scene_superballs_ms/recon_per_band_comparison.png": "file",
    "thesis/phase3_6_linear_recon_cave/figures/scene_superballs_ms/recon_rgb_pseudocolor_comparison.png": "file",
    "thesis/phase3_6_linear_recon_cave/data/scene_superballs_ms/recon_appendix_arrays.npz": "file",
    "thesis/phase3_6_linear_recon_cave/metrics/cave_recon_metrics.json": "file",
    "thesis/phase3_6_linear_recon_cave/metrics/cave_recon_summary.json": "file",
    "thesis/phase3_6_linear_recon_cave/reports/cave_recon_report.md": "file",
    "thesis/phase3_6_linear_recon_cave/metrics/reconstruction_metrics_summary.csv": "file",
    "thesis/reports/result_index.md": "file",
    "thesis/reports/figure_catalog.md": "file",
    "thesis/reports/repro_commands.md": "file",
    "thesis/reports/thesis_evidence_summary.md": "file",
    "thesis/reports/limitations.md": "file",
    "thesis/reports/lcd_forward_phase3_5_3_6_summary.md": "file",
}


def verify_handoff(release_root: Path) -> list[str]:
    errors = []

    for rel_path, expected_type in REQUIRED_STRUCTURE.items():
        target = release_root / rel_path
        if not target.exists():
            errors.append(f"MISSING: {rel_path}")
        elif expected_type == "file" and not target.is_file():
            errors.append(f"NOT A FILE: {rel_path}")

    manifest_path = release_root / "MANIFEST.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_files = set(manifest.get("files", {}).keys())
        for rel_path in REQUIRED_STRUCTURE:
            if rel_path in ("MANIFEST.json", "SHA256SUMS.txt"):
                continue
            if rel_path not in manifest_files:
                errors.append(f"NOT IN MANIFEST: {rel_path}")

    release_path = release_root / "RELEASE.json"
    if release_path.exists():
        with open(release_path) as f:
            release = json.load(f)
        required_release_keys = ["release_id", "source_repo", "source_commit", "lcd_forward_outputs"]
        for key in required_release_keys:
            if key not in release:
                errors.append(f"RELEASE.json missing key: {key}")

        outputs = release.get("lcd_forward_outputs", {})
        if not outputs.get("phase3_5_forward_validation"):
            errors.append("RELEASE.json: phase3_5_forward_validation not True")
        if not outputs.get("h_matrix_diagnostics"):
            errors.append("RELEASE.json: h_matrix_diagnostics not True")
        if not outputs.get("phase3_6_synthetic_reconstruction"):
            errors.append("RELEASE.json: phase3_6_synthetic_reconstruction not True")
        if not outputs.get("phase3_6_cave_reconstruction"):
            errors.append("RELEASE.json: phase3_6_cave_reconstruction not True")
        if not outputs.get("reconstruction_appendices"):
            errors.append("RELEASE.json: reconstruction_appendices not True")
        if outputs.get("real_target_capture"):
            errors.append("RELEASE.json: real_target_capture should be False")

    h_diag_path = release_root / "thesis" / "h_matrix_diagnostics" / "metrics" / "h_matrix_diagnostics.json"
    if h_diag_path.exists():
        with open(h_diag_path) as f:
            h = json.load(f)
        if h.get("n_full_rank_points", 0) == 0:
            errors.append("H diagnostics: zero full-rank points")
        if "median_condition_number" not in h:
            errors.append("H diagnostics: missing median_condition_number")

    cave_summary_path = release_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics" / "cave_recon_summary.json"
    if cave_summary_path.exists():
        with open(cave_summary_path) as f:
            cs = json.load(f)
        if not cs.get("scenes"):
            errors.append("CAVE summary: no scenes")
        for s in cs.get("scenes", []):
            if s.get("gain_db", 0) <= 0:
                errors.append(f"CAVE summary: {s.get('scene_id')} has non-positive gain")

    metrics_csv_path = release_root / "thesis" / "phase3_6_linear_recon_cave" / "metrics" / "reconstruction_metrics_summary.csv"
    if metrics_csv_path.exists():
        text = metrics_csv_path.read_text(encoding="utf-8")
        for required in ["synthetic_procedural", "cd_ms", "clay_ms", "superballs_ms", "multi_frame"]:
            if required not in text:
                errors.append(f"reconstruction_metrics_summary.csv missing: {required}")

    evidence_path = release_root / "thesis" / "reports" / "thesis_evidence_summary.md"
    if evidence_path.exists():
        text = evidence_path.read_text().lower()
        required_boundary_phrases = [
            "existence demonstration",
            "no sota",
            "not claimed",
            "not part of this release",
        ]
        missing = [p for p in required_boundary_phrases if p not in text]
        if missing:
            errors.append(f"thesis_evidence_summary.md missing boundary phrase(s): {missing}")

    release_path = release_root / "RELEASE.json"
    if release_path.exists():
        with open(release_path) as f:
            release = json.load(f)
        outputs = release.get("lcd_forward_outputs", {})
        if outputs.get("real_target_capture", True):
            errors.append("RELEASE.json: real_target_capture must be False")
        scope = release.get("scope", "").lower()
        if "existence" not in scope:
            errors.append("RELEASE.json scope must mention 'existence'")

    sums_path = release_root / "SHA256SUMS.txt"
    if sums_path.exists():
        with open(sums_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                expected_hash, rel_path = parts[0], " ".join(parts[1:])
                rel_path = rel_path.replace("\\", "/")
                target = release_root / rel_path
                if not target.exists():
                    errors.append(f"SHA256 target missing: {rel_path}")
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


def main():
    parser = argparse.ArgumentParser(description="Verify LCD_forward thesis handoff")
    parser.add_argument("release_root", help="Path to handoff release root")
    args = parser.parse_args()

    release_root = Path(args.release_root)
    if not release_root.exists():
        print(f"ERROR: release root not found: {release_root}")
        sys.exit(1)

    errors = verify_handoff(release_root)

    if errors:
        print(f"Handoff verification FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Handoff verification PASSED")
        print(f"  Release root: {release_root}")


if __name__ == "__main__":
    main()
