# LCD_forward Thesis Handoff Data Contract

## Identity

This release contains LCD_forward derived results (Phase 3.5¨C3.6 first pass).
It is NOT a hardware data release. Raw measured HDF5 remains in optic_system release.

## Input Source

- optic_system_phase3_release_20260520
  - train.h5 / val.h5 / test.h5
  - masks [N,1,1,64,64] uint8
  - psfs [N,1,3,512,512] float64
  - wavelengths_nm [450, 550, 650]
  - psf_roi_key: roi_512

## Output Structure

```
thesis/
  phase3_5_forward_validation/
    figures/    ¡ª measured vs predicted PSF, PCA basis
    metrics/    ¡ª psf_prediction_metrics.json
    reports/    ¡ª forward_validation_report.md
  h_matrix_diagnostics/
    figures/    ¡ª rank map, condition map, histogram, SV maps, OTF grid, CV map
    data/       ¡ª numpy arrays (.npy, .npz)
    metrics/    ¡ª h_matrix_diagnostics.json
    reports/    ¡ª h_matrix_diagnostics_report.md
  phase3_6_linear_recon_synthetic/
    figures/    ¡ª synthetic objects, masks, frames, single/multi recon
    metrics/    ¡ª reconstruction_metrics.json
    reports/    ¡ª linear_recon_report.md
  phase3_6_linear_recon_cave/
    figures/    ¡ª per-scene recon_comparison, rendered_frames
    metrics/    ¡ª cave_recon_metrics.json, cave_recon_summary.json
    reports/    ¡ª cave_recon_report.md
  reports/
    thesis_evidence_summary.md
    lcd_forward_phase3_5_3_6_summary.md
    limitations.md
provenance/
  lcd_forward_run_manifest.json
  bishe_first_pass.yaml
  phase3_6_debug_and_tuning.md
  optic_system_release_reference.json
```

## Metrics Interpretation

- forward validation: correlation > 0.95 indicates model captures PSF structure
- H matrix full-rank: encoding system is non-degenerate
- cave_recon: multi PSNR > single PSNR demonstrates multi-frame gain

## Not Included

- Real target capture
- Learned mask / GenerMask
- Deep neural reconstruction
- 512¡Á512 full-resolution reconstruction
- Multi-seed statistics

## Boundary

This release supports thesis existence proof. It does not claim:
- SOTA performance
- Optimal reconstruction
- Full physical model identification
