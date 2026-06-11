# Roadmap

This roadmap resets `LCD_forward` to the mainline LCD mask-to-operator modelling route.

It intentionally does not commit to final loaders, final external schemas, reconstruction baselines, or capture-control protocols before the relevant producer/consumer contracts exist.

## Phase 0: Documentation and Boundary Reset

- Mark the thesis-continuity prototype as legacy.
- Define mainline inputs and outputs.
- Add the cross-repository boundary document.
- Add handoff placeholder docs only.
- Preserve existing legacy tests and scripts unless explicitly scoped otherwise.

## Phase 1: Measured Evidence Contract Alignment

- Wait for `optic_system` measured evidence and PSF dictionary contracts to stabilize.
- Document expected evidence categories.
- Track needed inputs such as full-frame survey evidence, peak support reports, stability reports, layout profiles, and adaptive peak-cluster dictionaries.
- Do not implement fragile loaders against unstable raw paths.
- Do not define final external schemas without producer/consumer agreement.

## Phase 2: Peak-Cluster Operator Prototype

- Define internal representation for measured peak-cluster evidence.
- Prototype a sparse shift-patch forward operator.
- Implement adjoint consistency tests.
- Keep dense PSF materialization only for debugging and baseline comparison.
- Preserve compatibility with legacy dense PSF tests where practical.

## Phase 3: LCD-to-Operator Surrogate

- Consume `lcd_mask_families` mask identity and measured evidence.
- Learn or fit mask-to-peak-cluster response parameters.
- Validate against held-out mask specs and measured dictionaries.
- Keep mask-family generation in `lcd_mask_families`.
- Report uncertainty and provenance with fitted operator packages.

## Phase 4: Operator Diagnostics and Mask-Family Evaluation

- Generate OTF diagnostics.
- Generate H-matrix diagnostics.
- Evaluate mask families and mask sequences by operator quality, not PSF visual complexity.
- Produce mask-family evaluation reports.
- Propose capture plans back to `optic_system` where additional evidence would reduce operator uncertainty.

## Phase 5: Operator Handoff for Reconstruction

- Publish operator packages and diagnostics for downstream reconstruction.
- Include forward/adjoint API descriptions, mask sequence identity, and provenance.
- Keep reconstruction pipelines outside `LCD_forward`.
- Do not absorb learned reconstruction or target-scene inverse-problem evaluation into this repository.
