# LCD_forward Thesis Handoff Data Contract

## Identity

This release contains LCD_forward derived results (Phase 3.5-3.6 first pass).
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
    figures/    - measured vs predicted PSF, PCA basis
    metrics/    - psf_prediction_metrics.json
    reports/    - forward_validation_report.md
  h_matrix_diagnostics/
    figures/    - rank map, condition map, histogram, SV maps, OTF grid, CV map
    data/       - numpy arrays (.npy, .npz)
    metrics/    - h_matrix_diagnostics.json
    reports/    - h_matrix_diagnostics_report.md
  phase3_6_linear_recon_synthetic/
    figures/    - synthetic objects, masks, frames, single/multi recon, per-band/RGB comparisons
    data/       - recon_appendix_arrays.npz
    metrics/    - reconstruction_metrics.json
    reports/    - linear_recon_report.md
  phase3_6_linear_recon_cave/
    figures/    - per-scene per-band comparison, RGB pseudo-color comparison, rendered_frames
    data/       - per-scene recon_appendix_arrays.npz
    metrics/    - cave_recon_metrics.json, cave_recon_summary.json
    reports/    - cave_recon_report.md
  reports/
    thesis_evidence_summary.md
    result_index.md
    figure_catalog.md
    repro_commands.md
    lcd_forward_phase3_5_3_6_summary.md
    limitations.md
    metric_audit_response.md
    h_matrix_dc_otf_response.md
    solver_regularization_response.md
    alpha_interpretability.md
  alpha_sweep/
    cave_alpha_sweep_by_scene.csv
    cave_alpha_sweep_summary.csv
    cave_alpha_sweep.json / md
    alpha_psnr.png / pdf
    alpha_ssim.png / pdf
    alpha_display_ssim.png / pdf
provenance/
  lcd_forward_run_manifest.json
  bishe_first_pass.yaml
  phase3_6_debug_and_tuning.md
  optic_system_release_reference.json
```

## Metrics Interpretation

- forward validation: correlation > 0.95 indicates model captures PSF structure
- H matrix full-rank: encoding system is non-degenerate
- cave_recon: raw PSNR reports absolute amplitude error; interpret it together with per-band correlation, SSIM/audit metrics, and reconstruction figures
- recon_appendix_arrays.npz: GT object, single-frame recon, multi-frame recon, rendered frames, wavelengths, selected masks, and HDF5 provenance
- metric_audit_response.md: authoritative interpretation of visual-vs-PSNR mismatch; current complex128/alpha=3e-15 rerun removes the earlier clay_ms 450 nm negative-gain anomaly
- h_matrix_dc_otf_response.md: authoritative interpretation of OTF display subset and H-matrix DC rank behavior
- solver_regularization_response.md: authoritative explanation of precision-specific alpha, adaptive policy, and global-vs-frequency-scaled ridge behavior
- alpha_sweep/: 27-alpha sweep (1e-18 to 1e+0) with PSNR, raw SSIM, display-normalized SSIM, correlation, and PSNR gain per scene. Figures: alpha_psnr.png, alpha_ssim.png, alpha_display_ssim.png
- alpha_interpretability.md: authoritative explanation of mid-frequency encoding physics and DC singular-value collapse (condition ~3.3e7); explains why optimal alpha is 12 orders of magnitude smaller than typical Tikhonov regularizers

## Not Included

- Real target capture
- Learned mask / GenerMask
- Deep neural reconstruction
- 512x512 full-resolution reconstruction
- Multi-seed statistics

## Boundary

This release supports thesis existence proof. It does not claim:
- SOTA performance
- Optimal reconstruction
- Full physical model identification
