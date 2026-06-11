# LCD Mask-to-Operator Modelling

This repository owns LCD mask-to-peak-cluster/operator modelling for the mono-LCD programmable diffraction imaging system.

Mainline work should target measured-evidence consumption, peak-cluster/operator modelling, sparse shift-patch forward operators, matching adjoint operators, OTF diagnostics, H-matrix diagnostics, mask-family evaluation, and operator handoff packages.

## Current Mainline

Preferred repository route:

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

## Agent Rules

- Do target measured-evidence consumption, peak-cluster/operator modelling, forward/adjoint operators, operator diagnostics, and operator-oriented handoffs.
- Do treat reconstruction code as legacy or downstream-facing unless a user explicitly scopes reconstruction work.
- Do not add new reconstruction baselines as the default next step.
- Do not implement hardware control, camera control, LCD services, TLS services, acquisition control, synchronization, or raw capture logic.
- Do not duplicate `lcd_mask_families` mask-family generators, mask specs, projection policies, or core deterministic rendering logic.
- Do not define final external handoff schemas for `optic_system` or `reconstruction` before those producer/consumer contracts exist.
- Do not add loaders against unstable private paths from other repositories.
- Do preserve working legacy tests and scripts unless a change is explicitly scoped.

Allowed documentation and placeholder work:

- Markdown edits
- placeholder handoff directories and README files
- examples, manifests, or tiny illustrative fixtures when explicitly scoped
- legacy notes that clarify the thesis-continuity stack

## Repository Boundaries

- `lcd_mask_families` owns deterministic mask-generation functions, mask specs, mask sequence specs, family metadata, projection policies, and stable mask identity.
- `optic_system` owns hardware control, physical display, synchronized acquisition, raw HDF5 preservation, profile-aware measured artifacts, full-frame PSF surveys, diagnostics, and measured-evidence publication.
- `LCD_forward` owns LCD mask-to-peak-cluster/operator modelling and operator-oriented output packages.
- `reconstruction` owns reconstruction pipelines, learned reconstruction, inverse-problem evaluation, and target-scene reconstruction workflows.

See `docs/cross_repository_boundary.md` for the normative boundary.

## Legacy Prototype

The dense thesis-continuity prototype is retained for continuity, sanity checks, and regression tests. It includes:

- dense `mask -> PSF` learning
- `complex_field_basis`
- `psf_basis`
- PSF + object rendering
- simple reconstruction baseline
- synthetic sample dataset
- dense HDF5 tensor format

These components no longer define the mainline repository architecture. The interface pattern `forward_model(masks) -> psfs`, `render_frames(objects, psfs)`, and `recon_model(frames) -> objects_hat` is legacy-only unless explicitly scoped.

## Installation

```bash
pip install -e .
```
