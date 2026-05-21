# lcd_forward_phase3_5_3_6_release_20260520

LCD_forward Phase 3.5-3.6 first-pass thesis handoff.

Large binary payloads for this handoff are stored locally at `D:/datasets/LCD_forward/lcd_forward_phase3_5_3_6_release_20260520/`; Git stores only descriptors, reports, and provenance pointers.

## Quick Links

- First read: thesis/reports/thesis_evidence_summary.md
- Result index: thesis/reports/result_index.md
- Figure catalog: thesis/reports/figure_catalog.md
- Reproduction commands: thesis/reports/repro_commands.md
- Metric audit response: thesis/reports/metric_audit_response.md
- H-matrix DC/OTF response: thesis/reports/h_matrix_dc_otf_response.md
- Solver regularization response: thesis/reports/solver_regularization_response.md
- Alpha sweep appendix: thesis/alpha_sweep/cave_alpha_sweep.md
- Alpha sweep figures: thesis/alpha_sweep/alpha_psnr.png, alpha_ssim.png, alpha_display_ssim.png
- Alpha interpretability: thesis/reports/alpha_interpretability.md
- Thesis Chapter 4 figure guide: thesis/reports/downstream_thesis_figure_guide.md
- Thesis Chapter 4 figure package: thesis/thesis_figures/
- Data contract: data_contract.md
- Limitations: thesis/reports/limitations.md
- Debug record: provenance/phase3_6_debug_and_tuning.md

## Contents

1. Phase 3.5 forward validation (mask -> PSF prediction)
2. H-matrix frequency diagnostics (full-rank proof)
3. Phase 3.6 synthetic reconstruction (smoke test)
4. Phase 3.6 CAVE reconstruction (public dataset)
5. Reconstruction appendix: per-band figures, pseudo-RGB figures, and `.npz` arrays
6. Alpha sweep & interpretability: 27-alpha range (1e-18 to 1e+0), PSNR, SSIM raw, display SSIM analysis
7. Chapter 4 thesis-ready forward figures: PCA basis subset and measured-vs-predicted subset PDFs with CSV/manifest

## Input provenance

- optic_system Phase 3 release: see provenance/optic_system_release_reference.json
- CAVE public multispectral image database
