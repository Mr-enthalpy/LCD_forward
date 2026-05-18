# Mono LCD Spectral Prototype

This repository contains a prototype pipeline for mono-LCD-based programmable diffractive imaging.

Current scope:
- mono LCD only
- low-dimensional effective forward model
- mask -> PSF learning
- PSF + object -> frame rendering
- frame -> multispectral / multichannel reconstruction

Out of scope for the current prototype:
- RGB LCD
- hardware control
- camera ISP
- device synchronization
- full calibration automation
- full first-principles optical simulation

## Physical assumptions

The current prototype assumes:
- mono LCD
- non-coherent imaging
- paraxial approximation
- far-field approximation
- discrete wavelength bins
- low-dimensional effective forward model

These assumptions are part of the prototype definition and should not be changed implicitly.

## Repository structure

```text
src/
  datasets/     HDF5 datasets
  forward/      forward models and renderer
  recon/        reconstruction baseline
  losses/       forward/recon losses and metrics
  train/        train / validation loops
  utils/        seed and utility functions

scripts/
  prepare_sample_dataset.py
  train_forward.py
  train_recon.py
  eval_forward.py
  eval_recon.py
  smoke_test.py

tests/
  test_shapes.py
  test_forward_model.py
  test_renderer.py

```

## Data format

Training tensors use the following conventions.

Forward calibration:

- masks: [N, T, 1, Hm, Wm]
- psfs: [N, T, L, Hp, Wp]

Reconstruction:

- objects: [N, L, H, W]
- frames: [N, T, 1, H, W] (optional in HDF5; may be rendered online)
- masks: [N, T, 1, Hm, Wm]

Optional metadata:

- wavelengths: [L]
- spectral_response: [L]

A single .h5 file may contain both forward and reconstruction tensors.

## Core interfaces

Forward model:
```pycon
out = forward_model(masks)
psfs = out["psfs"]
```


Renderer:
```pycon
frames = render_frames(objects, psfs, spectral_response=None, noise_std=0.0)
```


Reconstruction model:
```pycon
out = recon_model(frames)
objects_hat = out["objects"]
```

## Forward models

Two forward-model families are currently supported.

### 1. complex_field_basis

Primary model.

Structure:

- low-resolution mask encoder
- low-dimensional complex coefficients
- complex field basis synthesis
- intensity projection to PSF

This is the preferred route because it preserves a low-dimensional amplitude-phase coupling model without explicitly reconstructing microscopic LCD geometry.

### 2. psf_basis

Baseline model.

Structure:

- low-resolution mask encoder
- low-dimensional coefficients
- direct low-rank PSF synthesis

This is a stable baseline and sanity-check path.

## Quick start

Install:
```Bash
pip install -e .
```

Generate sample data:
```Bash
python scripts/prepare_sample_dataset.py
```

Run smoke test:
```bash
python scripts/smoke_test.py
```

Train forward model:
```bash
python scripts/train_forward.py
```
Evaluate forward model:
```bash
python scripts/eval_forward.py
```
Train reconstruction baseline:
```bash
python scripts/train_recon.py
```
Evaluate reconstruction baseline:

```bash
python scripts/eval_recon.py
```

Run tests:
```bash
pytest -q
```

## Current development order

Recommended implementation / experimentation order:

1. sample dataset generation
2. single-wavelength single-frame forward training
3. held-out mask generalization check
4. renderer verification
5. reconstruction baseline
6. multi-wavelength extension
7. learnable masks / multiframe coding design
## Notes

The sample dataset generator is only a synthetic placeholder for engineering validation.
It should not be interpreted as a physically faithful mono-LCD forward simulator.

Real calibrated data should be converted into the same HDF5 tensor format so that the training and evaluation stack remains unchanged.