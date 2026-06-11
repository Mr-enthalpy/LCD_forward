# LCD Mask-to-Operator Modelling

`LCD_forward` is the LCD mask-to-operator modelling repository for the mono-LCD programmable diffraction imaging system.

The mainline responsibility is to consume mask identity and measured evidence, model the LCD mask-to-peak-cluster response, and publish operator-oriented outputs for downstream reconstruction.

## Mainline Route

The current repository direction is:

```text
MaskSpec / MaskSequenceSpec + measured evidence
  -> peak-cluster/operator model
  -> operator diagnostics
  -> operator handoff
```

Mainline inputs:

- `lcd_mask_families` `MaskInstanceSpec` and `MaskSequenceSpec`
- rendered or renderable LCD mask identity
- `optic_system` measured evidence handoffs
- measured PSF support evidence, full-frame survey evidence, peak support reports, stability reports, layout profiles, and adaptive peak-cluster dictionaries when those contracts exist

Mainline outputs:

- fitted or learned LCD-to-peak-cluster response models
- sparse peak-cluster / shift-patch forward operators
- matching adjoint operators
- OTF diagnostics, H-matrix diagnostics, and operator diagnostics
- mask-family evaluation reports
- operator-aware mask-sequence proposals
- `OperatorHandoff` packages for `reconstruction`
- future `CapturePlanHandoff` proposals for `optic_system`

## Repository Boundaries

`LCD_forward` does not own:

- hardware control, camera services, LCD services, or TLS services
- physical LCD embedding into real display buffers
- raw HDF5 capture or full-frame PSF survey acquisition
- mask-family definitions or core deterministic mask rendering logic
- reconstruction pipelines or learned reconstruction as a mainline responsibility
- target-scene inverse-problem evaluation
- final external handoff schemas before producer/consumer contracts exist

See `docs/cross_repository_boundary.md` for the normative cross-repository boundary.

## Legacy Thesis-Continuity Route

The existing dense prototype remains useful for continuity, sanity checks, and regression tests, but it no longer defines the mainline architecture.

Legacy components include:

- dense `mask -> PSF` learning
- `complex_field_basis` and `psf_basis` model variants
- dense PSF + object frame rendering
- simple reconstruction baseline
- synthetic sample dataset generation
- dense HDF5 tensor format

The legacy prototype interface is:

```pycon
out = forward_model(masks)
psfs = out["psfs"]
frames = render_frames(objects, psfs, spectral_response=None, noise_std=0.0)
out = recon_model(frames)
objects_hat = out["objects"]
```

This interface is a compatibility path only. It should not be treated as the main repository API.

See `docs/legacy_thesis_prototype.md` for details.

## Data Contracts

The dense HDF5 tensor format is a legacy / thesis-continuity format. It remains acceptable for baselines, compatibility, simple regression tests, and debugging dense PSF materialization.

Legacy dense tensors use:

- `masks`: `[N, T, 1, Hm, Wm]`
- `psfs`: `[N, T, L, Hp, Wp]`
- `objects`: `[N, L, H, W]`
- `frames`: `[N, T, 1, H, W]`
- optional `wavelengths`: `[L]`
- optional `spectral_response`: `[L]`

The mainline measured-evidence contract is peak-cluster/operator oriented. Real calibrated data should not be forced into the old dense HDF5 tensor contract as the only future path. Dense materialization may be used for compatibility and debugging, while mainline work should preserve measured evidence, peak-cluster evidence, adaptive peak-cluster dictionaries, operator packages, provenance, and diagnostics.

## Current Layout

```text
src/
  datasets/     legacy HDF5 dataset wrappers
  forward/      legacy dense forward models and renderer; future operator code may live here when scoped
  recon/        legacy reconstruction baseline, not the mainline owner
  losses/       forward/recon losses and metrics from the prototype
  train/        legacy train / validation loops
  utils/        seed and utility functions

scripts/
  prepare_sample_dataset.py
  train_forward.py
  train_recon.py
  eval_forward.py
  eval_recon.py
  smoke_test.py

docs/
  cross_repository_boundary.md
  roadmap.md
  legacy_thesis_prototype.md

handoffs/
  placeholder-only incoming and outgoing handoff areas
```

## Quick Start

Install:

```bash
pip install -e .
```

Run legacy smoke tests:

```bash
python scripts/smoke_test.py
pytest -q
```

The current scripts exercise the legacy thesis-continuity stack. They are retained to keep existing behavior reproducible while the mainline operator-modelling route is defined.
