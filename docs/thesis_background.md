# Thesis background

This thesis validates a mono-LCD programmable diffraction imaging prototype. The goal is not to outperform existing spectral imaging systems, but to demonstrate that a mono LCD can act as a stable programmable diffraction encoder whose mask-dependent PSFs can be measured, modelled at a simple level, and used in a minimal multiframe / multichannel reconstruction loop.

The hardware acquisition and calibration stages are handled by `optic_system`. `LCD_forward` is the backend that consumes `optic_system` HDF5 exports and performs forward validation, rendering, linear reconstruction, and thesis figure generation.

The thesis standard is feasibility and internal consistency. Complex neural reconstruction, learned mask design, full complex pupil reconstruction, and SOTA-level evaluation are explicitly outside the minimum scope.

## Core hypotheses

### Hypothesis 1: mono LCD as effective programmable diffraction encoder

Within the effective pupil window, changing the LCD mask stably changes the system PSF.

Evidence chain (from `optic_system`):
- Phase 3.1: effective pupil window is calibratable
- Phase 3.2a: same-mask PSF is repeatable
- Phase 3.2b: different-mask PSF difference is significantly larger than repeat noise

### Hypothesis 2: structured, low-dimensional, capturable LCD encoding features

The dOTF diagnostic reveals structured pupil-domain responses (abs/phase/real/imag) with low-dimensional, sparse, or banded features.

Evidence chain (from `optic_system`):
- Phase 3.3: dOTF diagnostic visualization

Note: dOTF in this thesis is a diagnostic visualization only, not a full complex pupil reconstruction.

### Hypothesis 3: measured mask-to-PSF data supports simple forward validation

Typical mask PSF differences are not random noise but can be explained by simple models or dictionary structure.

Evidence chain:
- Phase 3.4: measured PSF dictionary (from `optic_system`)
- Phase 3.5: simple forward baseline, measured vs predicted PSF comparison

### Hypothesis 4: multi-frame encoded observations contain multichannel-recoverable information

Multi-frame results outperform or differ from single-frame baselines, and basic channel structure is recoverable.

Evidence chain:
- Phase 3.6: measured PSF based renderer / forward operator, minimal linear reconstruction

## Success criteria (thesis standard)

- Multi-frame result is better than or distinguishable from single-frame baseline
- Basic channel structure is recoverable
- Figures and metrics support "prototype link established"
- No requirement for high-fidelity reconstruction, neural reconstruction, real complex scenes, or optimal mask design

## Not in scope

- RGB LCD main route
- learned mask / GenerMask design
- full complex pupil reconstruction
- neural reconstruction networks
- SOTA comparison
- end-to-end optimality
- complete physical first-principles modelling
- real complex scene recovery
