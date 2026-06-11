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

The dense prototype remains useful as historical context, but it no longer exists in the active source tree and no longer defines the mainline architecture.

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

This interface is historical only. It is preserved in Git history, not in the active package.

See `docs/legacy_thesis_prototype.md` for details.

## Data Contracts

The dense HDF5 tensor format is a legacy / thesis-continuity format. It is documented for historical context and possible future compatibility work, but it is not an active repository contract.

Legacy dense tensors use:

- `masks`: `[N, T, 1, Hm, Wm]`
- `psfs`: `[N, T, L, Hp, Wp]`
- `objects`: `[N, L, H, W]`
- `frames`: `[N, T, 1, H, W]`
- optional `wavelengths`: `[L]`
- optional `spectral_response`: `[L]`

The mainline measured-evidence contract is peak-cluster/operator oriented. Real calibrated data should not be forced into the old dense HDF5 tensor contract as the only future path. Dense materialization may be reintroduced only for explicitly scoped compatibility or debugging work, while mainline work should preserve measured evidence, peak-cluster evidence, adaptive peak-cluster dictionaries, operator packages, provenance, and diagnostics.

## Current Layout

```text
src/
  lcd_forward/
    masks/          placeholder for mask identity wrappers
    evidence/       placeholder for measured evidence representations
    peak_clusters/  placeholder for peak-cluster evidence and response parameters
    operators/      placeholder for sparse forward/adjoint operators
    surrogates/     placeholder for LCD-to-operator surrogate models
    diagnostics/    placeholder for OTF, H-matrix, and operator diagnostics
    handoffs/       placeholder for internal handoff helpers
    utils/          placeholder for small package utilities

scripts/
  README.md        placeholder only; no executable mainline scripts yet

docs/
  cross_repository_boundary.md
  roadmap.md
  legacy_thesis_prototype.md

handoffs/
  placeholder-only incoming and outgoing handoff areas

legacy/
  README.md        notes that legacy implementation lives in Git history

tests/
  test_package_import.py
```

## Quick Start

Install:

```bash
pip install -e .
```

Run the minimal package import test if `pytest` is available:

```bash
pytest -q
```

There are no executable mainline scripts yet. The active package contains architecture placeholders only; implementation should be added only when the corresponding measured-evidence, operator, surrogate, diagnostic, or handoff work is explicitly scoped.
