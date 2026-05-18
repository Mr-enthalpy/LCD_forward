# Agent constraints for LCD_forward thesis branch

This file defines mandatory constraints for any agent working on this repository.

## Thesis-closure constraints

1. Do not add hardware control code.
2. Do not import `optic_system` device services (camera, LCD, TLS).
3. Do not optimize for mainline generality after Phase 3.5.
4. Thesis goal is feasibility and internal consistency, not superiority.
5. Prefer direct scripts and readable reports.
6. Do not train complex models before simple baselines.
7. Always preserve data provenance.
8. Outputs must trace back to input HDF5 and config.

Phase 3.5/3.6 are thesis-closure stages. Prefer minimal, explicit, reproducible baselines over architectural elegance.

## Core assumptions

This thesis prototype assumes:
- mono LCD
- non-coherent imaging
- paraxial / far-field approximation
- discrete wavelength bins
- low-dimensional effective forward model instead of first-principles exact micro-geometry

## Repository structure

- `src/forward/`: forward models and renderer
- `src/recon/`: reconstruction network
- `src/losses/`: forward and reconstruction losses
- `src/datasets/`: HDF5 dataset wrappers
- `src/train/`: train / validation loops
- `scripts/`: dataset preparation, train / eval entry points, smoke test
- `tests/`: shape and module tests
- `configs/`: YAML configs

## Data format

Training tensors use the following conventions.

Forward calibration:
- `masks`: `[N, T, 1, Hm, Wm]`
- `psfs`: `[N, T, L, Hp, Wp]`

Reconstruction:
- `objects`: `[N, L, H, W]`
- `frames`: `[N, T, 1, H, W]` (optional in HDF5; can be rendered online)
- `masks`: `[N, T, 1, Hm, Wm]`

Optional metadata:
- `wavelengths`: `[L]`
- `spectral_response`: `[L]`

The sample and training format is HDF5.

## Forward model variants

Preferred order:
1. `complex_field_basis`
2. `psf_basis`

`complex_field_basis` is the main prototype route.
`psf_basis` is the baseline / control model.

## Thesis minimal success chain

```
1. Load optic_system export (train.h5 / val.h5 / test.h5)
2. Visualize measured PSF dictionary
3. Forward validation: simple baseline predicts PSF from mask
4. Render multi-frame observations via measured PSFs
5. Linear reconstruction: least squares / ridge / Tikhonov
6. Thesis figures: stable figures for report
```

## Priority order

```
data readable > figures reproducible > honest conclusions > model elegance
```

## Installation

```bash
pip install -e .
```
