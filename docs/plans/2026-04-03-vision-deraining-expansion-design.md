# Vision Deraining Expansion Design

**Goal**

Expand the existing synthetic denoising lesson with a stronger, more coherent
single-image deraining cluster by adding six compact-first deraining families:

- `ddn`
- `spanet`
- `did_mdn`
- `rcdnet`
- `transweather`
- `derainformer`

This work should stay inside the current denoising lesson instead of creating a
new task family, so users can keep using `--noise-type rain` with the same
training and discovery workflow.

---

## Scope

This feature targets **structured rain streak removal inside the existing
denoising / restoration lesson**.

Input:
- rainy image tensor with shape `(B, C, H, W)`

Output:
- restored image tensor with shape `(B, C, H, W)`

The implementation is educational and compact-first:
- no pretrained weights
- no external datasets
- no paper-exact reproduction requirements
- CPU-friendly tiny variants for smoke coverage

The purpose is to expose the main design patterns used in deraining:
- residual detail restoration
- spatial rain attention
- rain-density / decomposition conditioning
- iterative reconstruction
- transformer-based weather restoration

---

## Why This Stays in Lesson 10

`tracks/vision/lesson_10_synthetic_denoising` already provides:
- a synthetic image regression loop
- configurable `rain` noise generation
- architecture discovery commands
- CPU smoke coverage

Deraining is already introduced there through `jorder`, `rescan`, and
`prenet`. The missing piece is breadth: the lesson currently has only three
deraining-specific families, which is too narrow to show how the field evolved
from mask-guided CNNs to recurrent refinement and transformer restoration.

Keeping this work in Lesson 10 preserves the current user workflow:
- `python -m tracks.vision.lesson_10_synthetic_denoising.train --noise-type rain`
- `--list-arch`
- `--list-arch-families`
- `--arch <family>:<variant>`

No new lesson or new domain-specific zoo is required for v1.

---

## Package Layout

Create:
- `dlhub/vision/denoising/ddn.py`
- `dlhub/vision/denoising/spanet.py`
- `dlhub/vision/denoising/did_mdn.py`
- `dlhub/vision/denoising/rcdnet.py`
- `dlhub/vision/denoising/transweather.py`
- `dlhub/vision/denoising/derainformer.py`

Modify:
- `dlhub/vision/denoising/__init__.py`
- `tracks/vision/lesson_10_synthetic_denoising/model.py`
- `tracks/vision/lesson_10_synthetic_denoising/README.md`
- `tests/test_tracks_vision_denoising.py`

No changes are required to:
- `tracks/vision/lesson_10_synthetic_denoising/data.py`
- `tracks/vision/lesson_10_synthetic_denoising/train.py`

because rain synthesis, CLI argument parsing, and training mode support already
exist.

---

## Algorithm Cluster

Add six deraining families that cover distinct ideas.

### 1. `ddn`

Paper family:
- Deep Detail Network style deraining

Compact interpretation:
- shallow residual CNN
- derive a high-frequency detail branch from the rainy input
- predict rain residual and subtract it from the input

Teaching value:
- simplest "detail restoration" deraining baseline

### 2. `spanet`

Paper family:
- Spatial Attentive Network style deraining

Compact interpretation:
- residual CNN with spatial attention map
- attention map highlights rain-dominant regions before residual removal

Teaching value:
- shows explicit spatial localization without requiring recurrent state

### 3. `did_mdn`

Paper family:
- Density-aware Image Deraining using a multi-stream dense network

Compact interpretation:
- multi-branch dense CNN
- branch mixing acts as an implicit rain-density conditioner
- optionally expose an internal rain-density logits tensor, but final public
  output remains the restored tensor

Teaching value:
- introduces conditional deraining without changing the lesson API

### 4. `rcdnet`

Paper family:
- rain/background decomposition with iterative refinement

Compact interpretation:
- unrolled iterative block
- alternate between rain estimate and clean estimate refinement
- small fixed number of stages for CPU stability

Teaching value:
- shows decomposition / optimization-inspired deraining rather than pure
  feed-forward regression

### 5. `transweather`

Paper family:
- Transformer weather restoration

Compact interpretation:
- lightweight patch embedding + transformer blocks + convolutional head
- reuse existing padding helpers where spatial divisibility matters

Teaching value:
- first transformer-style deraining / bad weather restoration family in the
  lesson

### 6. `derainformer`

Paper family:
- deraining-focused transformer family

Compact interpretation:
- hybrid conv + token mixer / attention blocks
- emphasize local streak structure plus moderate global context

Teaching value:
- complements `transweather` with a more deraining-specific transformer flavor

---

## Model Contract

Each new file should follow the same denoising conventions already used in
`dlhub/vision/denoising/`.

Every family exposes:
- `_VARIANTS`
- `build_<family>_denoiser(...)`

Every builder accepts:
- `in_channels`
- `variant`

Optional family-specific internal hyperparameters should be hidden behind
variant specs unless the current denoising module style already exposes them.

Every model must:
- accept `(B, C, H, W)` tensors
- return a tensor with the same shape
- operate on `torch.float32`
- raise `ValueError` on invalid input rank or invalid channels

Variant policy:
- each family provides `_tiny`, `_small`, `_base`
- tiny variants must be CPU-smoke friendly on `32x32` and `64x64` inputs

---

## Integration Points

### `tracks/vision/lesson_10_synthetic_denoising/model.py`

This remains the main dispatch layer.

Required updates:
- import `_VARIANTS` for each new family in `list_supported_arches()`
- add each family to the returned arch list
- add `build_model()` dispatch branches
- add light aliases where useful, for example:
  - `did_mdn`, `did-mdn`, `didmdn`
  - `spanet`, `spa_net`
  - `rcdnet`, `rcd_net`
  - `derainformer`, `derain_former`
  - `transweather`, `trans_weather`

The existing `DenoiserAdapter` contract remains unchanged.

### `dlhub/vision/denoising/__init__.py`

Package exports should be updated so the new families are discoverable and the
module stays consistent with the rest of the directory.

### `tracks/vision/lesson_10_synthetic_denoising/README.md`

The deraining section should be expanded so users can discover the new cluster
from the lesson entrypoint rather than only from source code.

---

## Testing

Use the existing test file:
- `tests/test_tracks_vision_denoising.py`

Add three kinds of coverage.

### 1. Rain-data forward / backward smoke for new families

Add focused smoke coverage for:
- `ddn:ddn_tiny`
- `spanet:spanet_tiny`
- `did_mdn:did_mdn_tiny`
- `rcdnet:rcdnet_tiny`
- `transweather:transweather_tiny`
- `derainformer:derainformer_tiny`

Test shape contract:
- input rainy image `(B, C, H, W)`
- output restored image `(B, C, H, W)`
- finite loss against clean target
- backward pass succeeds

### 2. Arch discovery regression coverage

Update the existing CLI listing tests so they assert the new families appear in:
- `--list-arch-families`
- `--list-arch`
- `--list-arch --arch-family <family>`

This prevents implementation drift where modules exist but are not wired into
the lesson registry.

### 3. Existing lesson behavior stays stable

Do not remove or weaken current smoke coverage for:
- supervised denoising
- noise2noise
- blind-spot
- classical baselines
- rain noise argument parsing

The new tests should extend the current suite rather than replacing it.

---

## Documentation

Update the Lesson 10 README in four places:
- “目标” paragraph where the lesson scope is summarized
- “方法选择指南” deraining branch
- “模型一览” deraining subsection
- “快速参考” command examples

Recommended examples should include both CNN-style and transformer-style
derainers under `--noise-type rain`.

---

## Non-Goals for v1

Do not include in this iteration:
- a new standalone `deraining` package
- a new dedicated deraining lesson
- paired real rain datasets
- adversarial / perceptual / SSIM-heavy training changes
- video deraining
- rain accumulation / haze coupling beyond the current compact rain generator

This iteration optimizes for:
- stronger algorithm coverage inside the existing lesson
- low-friction learning workflow
- minimal surface-area changes
- easy extension for future restoration families
