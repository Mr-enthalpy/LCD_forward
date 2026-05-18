# Mono LCD Spectral Prototype

This repository contains a prototype pipeline for mono-LCD-based programmable diffractive imaging.

Current scope:
- mono LCD only
- low-dimensional effective forward model
- mask -> PSF learning
- PSF + object -> frame rendering
- frame -> multispectral / multichannel reconstruction

Out of scope for the current prototype:
- RGB LCD main route
- full hardware control
- camera ISP
- alignment / synchronization with real devices
- full physical calibration automation
- explicit high-resolution LCD micro-geometry as the main path

## Core assumptions

This prototype currently assumes:
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

## Installation

```bash
pip install -e .