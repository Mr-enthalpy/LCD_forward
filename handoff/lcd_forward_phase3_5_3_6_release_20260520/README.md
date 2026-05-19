# LCD_forward Phase 3.5–3.6 Thesis Handoff Descriptor

This directory contains the Git-tracked descriptor for the LCD_forward thesis handoff.
The large payload (figures, metrics, HDF5-referencing data) lives externally.

## External Payload Location

Canonical path (this workstation):
```
D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520/
```

Server path (when deployed):
```
/home/data2/<user>/lcd_forward_releases/lcd_forward_phase3_5_3_6_release_20260520/
```

## Verification

```bash
python scripts/verify_lcd_forward_handoff.py <payload_root>
```

## Structure

```
<payload_root>/
├── RELEASE.json              # Release metadata
├── MANIFEST.json             # File manifest with sizes
├── SHA256SUMS.txt            # SHA-256 checksums
├── data_contract.md          # Data format and interpretation
├── README.md                 # This file (duplicated in payload)
├── provenance/
│   ├── optic_system_release_reference.json
│   ├── lcd_forward_run_manifest.json
│   ├── bishe_first_pass.yaml
│   └── phase3_6_debug_and_tuning.md
└── thesis/
    ├── phase3_5_forward_validation/
    ├── h_matrix_diagnostics/
    ├── phase3_6_linear_recon_synthetic/
    ├── phase3_6_linear_recon_cave/
    └── reports/
```

## Git Policy

- This `handoff/` directory contains only the descriptor (this README)
- Payload files (.png, .json, .npy, .h5, etc.) live at the external path
- Never commit payload files to Git
