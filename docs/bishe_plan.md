# Thesis phase plan

This document defines the Phase 3.5 / 3.6 / 3.7 roadmap for `LCD_forward`. Phases 3.0.5b through 3.4 are owned by `optic_system` and are listed here for context only.

## Phase ownership

| Phase       | Title                                      | Owner        |
|-------------|--------------------------------------------|--------------|
| 3.0.5b      | PSF-safe camera parameters                 | optic_system |
| 3.1         | Effective pupil window calibration         | optic_system |
| 3.2a        | Camera-frame PSF ROI                       | optic_system |
| 3.2b        | PSF repeatability & mask-induced diversity | optic_system |
| 3.3         | dOTF diagnostic visualization              | optic_system |
| 3.4         | Measured PSF dictionary capture & HDF5 export | optic_system |
| **3.5**     | **Measured PSF forward validation**        | **LCD_forward** |
| **3.6**     | **Minimal multiframe / multichannel reconstruction** | **LCD_forward** |
| **3.7**     | **Thesis figures and report freeze**       | **LCD_forward** |

## Phase 3.5 — Measured PSF forward validation

### Goal

Validate that measured mask-to-PSF data can support a simple forward explanation of typical LCD mask-induced PSF differences. Not training a high-performance model.

### Input

```
data/optic_system/psf_dictionary/train.h5
data/optic_system/psf_dictionary/val.h5
data/optic_system/psf_dictionary/test.h5
```

### Minimum methods (by priority)

1. **Baseline A**: dictionary lookup / nearest-neighbor by mask_id — sanity check
2. **Baseline B**: low-dimensional PSF PCA basis + mask-to-PCA-coefficient ridge regression or small MLP
3. **Baseline C**: existing `psf_basis` or `complex_field_basis` surrogates — only if the existing prototype code is easy to reuse

Do not require complex-field model as the first step.

### Outputs

```
outputs/forward_validation/
    measured_vs_predicted_psf.png
    psf_error_metrics.json
    psf_basis_preview.png
    representative_cases/
    forward_validation_report.md
```

### Success criteria

1. Can read `optic_system` exported HDF5
2. Can display and inspect measured PSFs
3. Can produce baseline predictions for test masks
4. Can output MSE / relative error / correlation metrics
5. Typical mask predicted-vs-measured figures are thesis-ready
6. Error is recorded honestly; no requirement to inflate performance

## Phase 3.6 — Minimal multiframe / multichannel reconstruction

### Goal

Demonstrate that multi-frame encoded observations contain enough information for a simple multichannel / multispectral recovery. Not building an optimal reconstruction network.

### Input modes

#### A. Synthetic observations from measured PSFs (preferred for speed)

- objects: synthetic simple targets
- psfs: measured dictionary PSFs
- frames: rendered by `LCD_forward`

#### B. Real target captures from `optic_system`

If `optic_system` later provides real target captures:
- target_raw_h5 from `optic_system`
- selected mask sequence
- measured PSF dictionary

Real target capture remains owned by `optic_system`.

### Minimum methods

1. Construct convolution-style forward operator
2. Select a few masks from the measured PSF dictionary
3. Generate or read multi-frame observations
4. Linear reconstruction: least squares / ridge / Tikhonov / NNLS
5. Output single-frame vs multi-frame comparison

Do not start with deep reconstruction networks.

### Outputs

```
outputs/linear_recon/
    rendered_frames.npy
    recon_single_frame.npy
    recon_multiframe.npy
    recon_comparison.png
    reconstruction_metrics.json
    linear_recon_report.md
```

### Success criteria

1. Forward operator constructible
2. Multi-frame observations generatable or readable
3. Linear reconstruction runs successfully
4. Output figures show basic channel / structure recovery
5. No requirement to outperform existing systems or optimize masks

## Phase 3.7 — Thesis figures and report freeze

### Goal

Aggregate figures and metrics from all phases into a consistent thesis-ready set.

### Source phases

- optic_system Phase 3.1: effective pupil geometry
- optic_system Phase 3.2: PSF ROI and repeatability
- optic_system Phase 3.3: dOTF diagnostics
- optic_system Phase 3.4: measured PSF dictionary
- LCD_forward Phase 3.5: forward validation
- LCD_forward Phase 3.6: reconstruction

### Outputs

```
outputs/bishe_figures/
    system_pipeline.png
    effective_pupil_window.png
    psf_repeatability_summary.png
    dotf_diagnostic.png
    psf_dictionary_examples.png
    forward_validation_examples.png
    reconstruction_result.png
    figure_manifest.json
```
