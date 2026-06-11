# Legacy Thesis-Continuity Prototype

The old dense PSF prototype reflected the thesis-continuity branch of `LCD_forward`. It has been removed from the active source tree and is preserved in Git history.

It remains useful as historical context for continuity, sanity checks, regression-test design, and simple engineering validation, but it no longer defines the mainline repository architecture.

## Included Components

The legacy prototype includes:

- dense mask-to-PSF learning
- `complex_field_basis`
- `psf_basis`
- PSF + object rendering
- simple reconstruction baseline
- synthetic sample dataset generation
- dense HDF5 tensor format

## Legacy Interfaces

The legacy interface pattern was:

```pycon
out = forward_model(masks)
psfs = out["psfs"]
frames = render_frames(objects, psfs, spectral_response=None, noise_std=0.0)
out = recon_model(frames)
objects_hat = out["objects"]
```

This pattern is not present in the active package. It should not be presented as the main `LCD_forward` API.

## Legacy Dense HDF5 Format

Legacy training tensors use:

- `masks`: `[N, T, 1, Hm, Wm]`
- `psfs`: `[N, T, L, Hp, Wp]`
- `objects`: `[N, L, H, W]`
- `frames`: `[N, T, 1, H, W]`
- optional `wavelengths`: `[L]`
- optional `spectral_response`: `[L]`

The dense HDF5 tensor format is not the long-term universal data contract. It may be referenced for future compatibility tests or debugging dense materializations, but it is not an active repository contract.

## Model Variants

### `complex_field_basis`

The legacy primary model uses:

- low-resolution mask encoding
- low-dimensional complex coefficients
- complex field basis synthesis
- intensity projection to dense PSF tensors

### `psf_basis`

The legacy baseline model uses:

- low-resolution mask encoding
- low-dimensional coefficients
- direct low-rank dense PSF synthesis

## Current Status

Legacy scripts such as `scripts/train_forward.py`, `scripts/eval_forward.py`, `scripts/train_recon.py`, and `scripts/eval_recon.py` previously exercised the dense prototype stack. They have been removed from the active tree and are preserved in Git history.

New mainline work should target measured evidence, peak-cluster evidence, adaptive peak-cluster dictionaries, sparse shift-patch operators, forward/adjoint operator consistency, OTF diagnostics, H-matrix diagnostics, mask-family evaluation, and operator handoff packages.

Real calibrated data should not be forced into this old dense HDF5 layout as the only future path. Dense PSF tensors may be generated as compatibility or debugging artifacts, while the mainline measured-evidence contract should remain peak-cluster/operator oriented.
