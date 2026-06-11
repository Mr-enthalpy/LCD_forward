# LCD_forward Skill

This skill defines how agents should work inside `LCD_forward`.

`LCD_forward` is the LCD mask-to-operator modelling repository for the mono-LCD programmable diffraction imaging system.

The active mainline is:

```text
MaskSpec / MaskSequenceSpec + measured evidence
  -> peak-cluster/operator model
  -> operator diagnostics
  -> operator handoff
```

The repository is not a hardware-control repository, not a mask-family definition repository, and not a reconstruction repository.

## Required Reading Order

Before making changes, read these files in order:

1. `README.md`
2. `AGENTS.md`
3. `docs/cross_repository_boundary.md`
4. `docs/roadmap.md`
5. `docs/legacy_thesis_prototype.md`
6. Relevant `src/lcd_forward/**/README.md`
7. Relevant `handoffs/**/README.md`

Do not start from old source paths, old scripts, or old tensor conventions. The dense thesis prototype is historical unless explicitly scoped.

## Repository Role

`LCD_forward` owns:

```text
LCD mask-to-peak-cluster/operator modelling
measured-evidence consumption
peak-cluster/operator internal representation
sparse shift-patch forward/adjoint operators
LCD-to-operator surrogate models
OTF / H-matrix / operator diagnostics
mask-family and mask-sequence evaluation
operator handoff publication
capture-plan proposal generation
```

`LCD_forward` consumes:

```text
lcd_mask_families MaskInstanceSpec / MaskSequenceSpec
lcd_mask_families mask identity and hash provenance
optic_system measured evidence handoffs
full-frame PSF survey evidence
peak support reports
support stability reports
layout profiles
adaptive peak-cluster dictionaries
```

`LCD_forward` produces:

```text
operator packages
forward/adjoint API descriptions
OTF diagnostics
H-matrix diagnostics
mask-family evaluation reports
operator-aware mask-sequence proposals
future capture-plan proposals
```

## Explicit Non-Ownership

Do not add or restore ownership of:

```text
camera control
LCD display services
TLS control
device synchronization
raw HDF5 acquisition
full-frame survey acquisition
mask-family core rendering
mask-family hash rules
physical LCD embedding into display buffers
reconstruction pipelines
learned reconstruction
target-scene inverse-problem evaluation
external final handoff schemas before contracts exist
```

If a task requires one of these, stop and keep the work at the boundary document or placeholder level unless explicitly instructed otherwise.

## Active Source Shape

The active package is:

```text
src/lcd_forward/
  masks/
  evidence/
  peak_clusters/
  operators/
  surrogates/
  diagnostics/
  handoffs/
  utils/
```

Each directory is a mainline architecture slot.

Current placeholder rule:

```text
README.md
empty __init__.py
.gitkeep where needed
```

Do not refill a placeholder directory with speculative implementation. Add code only when the task explicitly scopes that layer.

## Directory Semantics

### `src/lcd_forward/masks/`

Future local wrappers around `lcd_mask_families` mask identity.

Allowed future responsibilities:

```text
read already-published mask identity
preserve mask spec/hash provenance
adapt mask identity for LCD_forward internal records
```

Forbidden:

```text
reimplement mask-family renderers
fork projection policies
redefine hash semantics
copy lcd_mask_families family logic
```

### `src/lcd_forward/evidence/`

Future representation of measured evidence from `optic_system`.

Allowed future responsibilities:

```text
internal measured-evidence records
artifact-reference records
validated manifest-level ingestion after contracts stabilize
```

Forbidden:

```text
read optic_system private raw paths
treat raw HDF5 layout as external API
define final optic_system schema unilaterally
perform hardware acquisition
```

### `src/lcd_forward/peak_clusters/`

Future peak-cluster evidence and response-parameter representation.

Allowed future responsibilities:

```text
peak slot metadata
local patch metadata
offset / support / stability records
response parameter containers
```

Forbidden before explicit scope:

```text
cluster detection implementation
optic_system artifact parser
adaptive layout inference
final dictionary schema
```

### `src/lcd_forward/operators/`

Future sparse shift-patch operator implementation.

Allowed future responsibilities:

```text
forward operator
adjoint operator
boundary policy
dense materialization only for debug
operator package assembly
```

Forbidden before explicit scope:

```text
actual shift-patch implementation
adjoint implementation
dense PSF renderer restoration
reconstruction solver
```

### `src/lcd_forward/surrogates/`

Future LCD mask-to-operator surrogate models.

Allowed future responsibilities:

```text
mask-to-peak-cluster parameter surrogate
operator-parameter surrogate
uncertainty estimates
held-out mask validation hooks
```

Forbidden before explicit scope:

```text
neural network code
training loop
torch dependency
optimizer
loss suite
```

### `src/lcd_forward/diagnostics/`

Future operator diagnostics.

Allowed future responsibilities:

```text
OTF diagnostics
H-matrix diagnostics
conditioning summaries
mask-family evaluation
operator-quality reports
```

Forbidden before explicit scope:

```text
diagnostic implementation
plotting stack
benchmark suite
final report schema
```

### `src/lcd_forward/handoffs/`

Future internal helpers for outgoing handoffs.

Allowed future responsibilities:

```text
operator handoff records
capture-plan proposal records
manifest references
provenance summaries
```

Forbidden before explicit scope:

```text
final external schema
external repo client
direct reconstruction integration
direct optic_system capture control
```

### `src/lcd_forward/utils/`

Small shared utilities only.

Do not place domain logic here. If a utility grows domain semantics, move it into the appropriate mainline directory.

## Legacy Policy

The old thesis-continuity prototype included:

```text
dense mask-to-PSF learning
complex_field_basis
psf_basis
PSF + object rendering
simple reconstruction baseline
synthetic sample dataset
dense HDF5 tensor format
```

It is preserved in Git history and documented in `docs/legacy_thesis_prototype.md`.

Do not restore legacy implementation unless explicitly scoped as compatibility work.

Do not present legacy interfaces as current APIs:

```text
forward_model(masks) -> psfs
render_frames(objects, psfs)
recon_model(frames) -> objects_hat
```

Do not force real calibrated data into the old dense HDF5 contract as the only future path.

Dense materialization may be reintroduced only for explicitly scoped debugging, compatibility, or baseline comparison.

## Dependency Policy

The empty architecture must not require heavy or external dependencies.

Do not add these as hard dependencies unless explicitly scoped:

```text
torch
h5py
matplotlib
opencv
optic_system
lcd_mask_families
reconstruction
```

Prefer documenting a future dependency over adding it prematurely.

If a task needs `lcd_mask_families`, use it as a boundary dependency only after the corresponding wrapper task is explicitly scoped. Do not copy its implementation.

## Handoff Policy

Handoff directories are placeholders unless a task explicitly scopes manifest work.

General rules:

```text
no large data in Git
no final external schema commitment
no external repository imports
no producer/consumer protocol implementation without agreement
docs and tiny examples only unless explicitly scoped
```

Incoming handoffs:

```text
handoffs/incoming/mask_specs/
handoffs/incoming/measured_evidence/
```

Outgoing handoffs:

```text
handoffs/outgoing/operator_handoffs/
handoffs/outgoing/capture_plan_proposals/
```

Use handoff documents to clarify boundaries, not to smuggle in implementation.

## Roadmap Gate Discipline

Follow the roadmap gates.

### Phase 0: Boundary and architecture reset

Allowed:

```text
documentation
placeholder directories
package import check
legacy deletion / demotion
```

Forbidden:

```text
operator implementation
loader implementation
surrogate implementation
diagnostic implementation
```

### Phase 1: Measured evidence contract alignment

Allowed:

```text
document expected evidence categories
draft internal record notes
track missing upstream contracts
```

Forbidden:

```text
fragile loaders against optic_system private paths
final schemas
raw HDF5 assumptions as stable APIs
```

### Phase 2: Peak-cluster operator prototype

Allowed only when explicitly scoped.

Expected direction:

```text
minimal internal peak-cluster representation
sparse shift-patch forward prototype
adjoint consistency tests
debug dense materialization only if scoped
```

### Phase 3: LCD-to-operator surrogate

Allowed only after measured evidence and internal operator representation are stable enough.

Expected direction:

```text
mask identity + measured evidence -> operator parameters
held-out mask validation
uncertainty and provenance
```

### Phase 4: Diagnostics and mask-family evaluation

Allowed only after operator representation exists.

Expected direction:

```text
OTF diagnostics
H-matrix diagnostics
conditioning summaries
operator-quality ranking
mask-sequence proposals
```

### Phase 5: Operator handoff

Allowed only after downstream reconstruction expectations are known.

Expected direction:

```text
operator package
forward/adjoint API description
diagnostics
mask identity
provenance
```

## Testing Policy

At the placeholder stage, tests should be minimal.

Allowed:

```text
import lcd_forward
check package structure if needed
markdown / path consistency checks if explicitly scoped
```

Do not write tests that imply unimplemented functionality exists.

Forbidden before implementation scope:

```text
operator tests
adjoint tests
loader tests
surrogate tests
H-matrix tests
reconstruction tests
external repo integration tests
```

When implementation begins, each module should add tests matching its actual contract. Do not test future designs as if they already exist.

## Documentation Policy

Update documentation whenever the active architecture changes.

Keep these documents consistent:

```text
README.md
AGENTS.md
docs/cross_repository_boundary.md
docs/roadmap.md
docs/legacy_thesis_prototype.md
src/lcd_forward/**/README.md
handoffs/**/README.md
```

Preferred terms:

```text
measured evidence
peak-cluster evidence
adaptive peak-cluster dictionary
shift-patch operator
forward/adjoint operator
OTF diagnostics
H-matrix diagnostics
operator package
operator handoff
mask-family evaluation
mask-sequence proposal
capture-plan proposal
```

Legacy-only terms unless clearly marked:

```text
dense PSF tensor contract
training HDF5
reconstruction baseline
recon_model
train_recon.py
PSF renderer as central API
forward_model(masks) -> psfs as main interface
```

## PR Review Checklist

A PR is acceptable only if it satisfies its declared phase.

Check:

```text
Does it preserve LCD_forward as mask-to-operator modelling?
Does it avoid hardware-control responsibility?
Does it avoid reconstruction ownership?
Does it avoid duplicating lcd_mask_families?
Does it avoid unstable optic_system private paths?
Does it avoid final schemas before contracts exist?
Does it avoid adding heavy dependencies prematurely?
Does it keep placeholders empty unless implementation is explicitly scoped?
Does it update docs when architecture changes?
Does it include only tests matching implemented behavior?
```

Reject or revise PRs that quietly restore the old thesis prototype as the active architecture.

## Short Rule

When unsure, preserve the boundary and leave a placeholder.

Do not implement an interface merely because a future directory exists.
