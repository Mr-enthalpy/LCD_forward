"""Update handoff docs: change only CURRENT/CONFIGURED alpha references from 5e-15 to 3e-15.
Does NOT touch sweep data tables or historical range discussions."""
from pathlib import Path
import json

H = Path(r"D:\datasets\LCD_forward\lcd_forward_phase3_5_3_6_release_20260520")

# File-specific targeted replacements
patches = [
    # cave_alpha_sweep.md
    (H / "thesis" / "alpha_sweep" / "cave_alpha_sweep.md", [
        ("Current alpha: `5e-15`", "Current alpha: `3e-15`"),
    ]),

    # data_contract.md
    (H / "data_contract.md", [
        ("current complex128/alpha=5e-15 rerun", "current complex128/alpha=3e-15 rerun"),
    ]),

    # thesis_evidence_summary.md
    (H / "thesis" / "reports" / "thesis_evidence_summary.md", [
        ("configured handoff value is `5e-15`", "configured handoff value is `3e-15`"),
    ]),

    # metric_audit_response.md
    (H / "thesis" / "reports" / "metric_audit_response.md", [
        ("ridge_alpha: 5.0e-15", "ridge_alpha: 3.0e-15"),
        ("complex128/alpha=5e-15", "complex128/alpha=3e-15"),
    ]),

    # phase3_6_cave_metrics_response.md
    (H / "thesis" / "reports" / "phase3_6_cave_metrics_response.md", [
        ("ridge_alpha: 5.0e-15", "ridge_alpha: 3.0e-15"),
        ("complex128/alpha=5e-15", "complex128/alpha=3e-15"),
    ]),

    # solver_regularization_response.md
    (H / "thesis" / "reports" / "solver_regularization_response.md", [
        ("ridge_alpha: 5.0e-15", "ridge_alpha: 3.0e-15"),
        (chr(0x60) + "alpha=5e-15" + chr(0x60) + " is the best tested value",
         chr(0x60) + "alpha=3e-15" + chr(0x60) + " is the best tested value"),
        ("alpha=1e-6" + chr(0x60) + " and " + chr(0x60) + "alpha=5e-15" + chr(0x60),
         "alpha=1e-6" + chr(0x60) + " and " + chr(0x60) + "alpha=3e-15" + chr(0x60)),
        ("best tested CAVE value became " + chr(0x60) + "alpha=5e-15" + chr(0x60),
         "best tested CAVE value became " + chr(0x60) + "alpha=3e-15" + chr(0x60)),
    ]),

    # phase3_6_solver_regularization_response.md (backup copy)
    (H / "thesis" / "reports" / "phase3_6_solver_regularization_response.md", [
        ("ridge_alpha: 5.0e-15", "ridge_alpha: 3.0e-15"),
        (chr(0x60) + "alpha=5e-15" + chr(0x60) + " is the best tested value",
         chr(0x60) + "alpha=3e-15" + chr(0x60) + " is the best tested value"),
        ("alpha=1e-6" + chr(0x60) + " and " + chr(0x60) + "alpha=5e-15" + chr(0x60),
         "alpha=1e-6" + chr(0x60) + " and " + chr(0x60) + "alpha=3e-15" + chr(0x60)),
        ("best tested CAVE value became " + chr(0x60) + "alpha=5e-15" + chr(0x60),
         "best tested CAVE value became " + chr(0x60) + "alpha=3e-15" + chr(0x60)),
    ]),

    # cave_recon_report.md
    (H / "thesis" / "phase3_6_linear_recon_cave" / "reports" / "cave_recon_report.md", [
        ("alpha=5e-15", "alpha=3e-15"),
        ("`alpha=5e-15`", "`alpha=3e-15`"),
    ]),

    # linear_recon_report.md (synthetic)
    (H / "thesis" / "phase3_6_linear_recon_synthetic" / "reports" / "linear_recon_report.md", [
        ("alpha=5e-15", "alpha=3e-15"),
    ]),
]

count = 0
for path, replacements in patches:
    if not path.exists():
        print(f"  SKIP (not found): {path.relative_to(H)}")
        continue
    text = path.read_text(encoding="utf-8")
    for old, new in replacements:
        if old in text:
            text = text.replace(old, new)
            count += 1
        else:
            print(f"  NOTE: pattern not found in {path.relative_to(H)}: {old[:60]}")
    path.write_text(text, encoding="utf-8")

# Update alpha_sweep.json
asj = H / "thesis" / "alpha_sweep" / "cave_alpha_sweep.json"
with open(asj) as f:
    d = json.load(f)
d["current_alpha"] = 3e-15
d["current_alpha_label"] = "3e-15"
with open(asj, "w") as f:
    json.dump(d, f, indent=2)
count += 1

print(f"\n{count} replacements applied across {len(patches)} files + alpha_sweep.json.")

# Scan for any remaining "5e-15" as current/configured (not in sweep tables)
import re
for f in sorted(H.rglob("*.md")):
    text = f.read_text(encoding="utf-8")
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        # Skip sweep data table rows (lines starting with | and containing -15)
        if stripped.startswith("|") and "e-15" in stripped:
            continue
        if "5e-15" in stripped and "current" in stripped.lower():
            print(f"  REVIEW: {f.relative_to(H)}:{i}: {stripped[:100]}")
