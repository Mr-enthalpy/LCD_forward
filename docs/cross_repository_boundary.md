# Cross-Repository Boundary

This document defines the normative boundary for `LCD_forward` within the mono-LCD programmable diffraction imaging system.

`LCD_forward` owns LCD mask-to-peak-cluster/operator modelling. It consumes mask identity and measured evidence, fits or learns operator-oriented response models, and publishes operator packages and diagnostics for downstream use.

## `lcd_mask_families -> LCD_forward`

Meaning:

- `MaskInstanceSpec`
- `MaskSequenceSpec`
- family metadata
- rendered or renderable mask identity
- projection policy
- deterministic hashes and stable mask identifiers

`LCD_forward` may:

- wrap mask specs for differentiable use
- encode mask identity for surrogate training
- evaluate mask families and mask sequences by operator quality
- preserve provenance linking operator outputs back to mask specs and hashes

`LCD_forward` must not:

- redefine core mask-family renderers
- duplicate deterministic mask-generation logic
- fork projection policies or stable identity rules
- treat local mask wrappers as the source of truth for mask families

## `optic_system -> LCD_forward`

Meaning:

- `MeasuredEvidenceHandoff`
- artifact manifests or references to measured support/operator evidence
- `FullFramePSFSurvey` evidence
- `SensorEnergyCenterProfile` evidence
- peak support reports
- stability reports
- layout profiles
- adaptive peak-cluster dictionaries

`LCD_forward` may:

- consume published measured evidence handoffs
- fit or learn LCD-to-peak-cluster response models from measured evidence
- derive sparse shift-patch operators from measured peak-cluster evidence
- report missing evidence categories needed for better operator modelling

`LCD_forward` must not:

- own hardware control or synchronized acquisition
- read unstable private raw-capture internals from `optic_system`
- assume raw HDF5 layout details are stable external APIs
- implement full-frame PSF survey acquisition
- treat measured-evidence placeholder docs as final schemas

## `LCD_forward -> reconstruction`

Meaning:

- `OperatorHandoff`
- fitted or learned operator packages
- forward/adjoint API descriptions
- sparse peak-cluster / shift-patch operator metadata
- OTF diagnostics
- H-matrix diagnostics
- mask sequence identity
- evidence provenance and model provenance

`LCD_forward` may:

- publish operator packages for downstream reconstruction
- describe forward and adjoint operator semantics
- include diagnostics needed to evaluate operator conditioning and coverage
- provide compatibility dense PSF materializations for debugging when scoped

`LCD_forward` does not own:

- reconstruction pipelines
- learned reconstruction as a mainline responsibility
- target-scene inverse-problem evaluation
- final downstream schemas before `reconstruction` contract alignment exists

## `LCD_forward -> optic_system`

Meaning:

- future `CapturePlanHandoff` proposals
- requested mask sequences
- wavelength sweeps
- repeat counts
- uncertainty-reduction captures
- diagnostic capture requests motivated by operator modelling gaps

These are future handoff concepts, not implemented APIs unless explicitly added later.

`LCD_forward` may propose:

- mask-sequence proposals for measured evidence improvement
- capture-plan proposals to reduce operator uncertainty
- evidence requests tied to peak-cluster support, stability, or layout ambiguity

`LCD_forward` must not:

- implement capture-control protocols
- bypass `optic_system` hardware orchestration
- assume proposal documents are executable acquisition plans
- define final capture-plan schemas before producer/consumer agreement exists
