# Vision Super-Resolution Design

**Goal**

Add a local, compact-first super-resolution algorithm family to DL-Hub with:
- local super-resolution model families
- a unified zoo + CLI
- fast smoke tests that run on CPU with no downloads
- one synthetic paired-supervision Vision lesson

This should follow the repository patterns used by:
- `dlhub.vision.style_transfer_zoo`
- `dlhub.vision.video_summarization_zoo`
- `tracks/vision/lesson_10_synthetic_denoising`

---

## Scope

This feature targets image super-resolution as a dedicated task family rather than
reusing denoising models or interfaces.

Input:
- low-resolution image tensor with shape `(B, C, H, W)`

Output:
- `sr`: super-resolved image tensor with shape `(B, C, H * scale, W * scale)`

Optional outputs may include:
- `residual`
- `features`
- `attention_map`

The family is meant to teach how classical CNN SR, residual-channel attention SR,
dense residual SR, and lightweight transformer SR differ at a high level, not to
exactly reproduce benchmark implementations.

---

## Package Layout

Add:
- `dlhub/vision/super_resolution/`
- `dlhub/vision/super_resolution_zoo.py`
- `scripts/super_resolution_zoo.py`
- `tests/test_dlhub_vision_super_resolution_zoo.py`

Add one lesson:
- `tracks/vision/lesson_17_synthetic_super_resolution/`
- `tests/test_tracks_vision_super_resolution.py`

Update:
- `tracks/vision/README.md`

Initial local families:
- `srcnn`
- `fsrcnn`
- `edsr_sr`
- `rcan_sr`
- `rdn_sr`
- `swinir_sr`

Each family provides three variants:
- `*_tiny`
- `*_small`
- `*_base`

Zoo prefix:
- `sr:<variant>`

Current initial coverage:
- 6 families
- 18 arches

---

## Model Contract

Every family exposes:
- `_VARIANTS`
- `build_<family>_super_resolver(...)`

Every model supports:
- `model(low_res)` where `low_res` is `(B, C, H, W)`

Every model returns a dict with:
- `sr`

Unified builder entry point:
- `build_local_model(arch_id, *, in_channels, upscale_factor, image_size, width_mult, dropout, **kwargs)`

Initial support policy:
- `upscale_factor=2` is the only officially supported factor in v1
- builders may accept the parameter but must reject unsupported factors explicitly

---

## Zoo and CLI

`dlhub.vision.super_resolution_zoo` exposes:
- `list_local_arches() -> list[str]`
- `build_local_model(...) -> nn.Module`

The zoo should use AST-based lazy discovery, mirroring the existing local zoo
pattern already used by recent vision task families.

`scripts/super_resolution_zoo.py` should support:
- `--list`
- `--search`
- `--limit`
- `--smoke`

Smoke mode should:
- create random low-resolution input
- build the requested arch
- run a forward pass
- print a short summary of the model and output structure

---

## Lesson Design

Add:
- `tracks/vision/lesson_17_synthetic_super_resolution/`

The lesson should be compact-first, CPU-friendly, and self-contained.

Training formulation:
- create synthetic HR images online
- apply a simple degradation pipeline to generate LR inputs
- train a super-resolution model on paired `(lr, hr)` samples

Recommended degradation pipeline:
- mild blur
- bicubic or bilinear downsampling by factor `2`
- optional light noise or compression-like bias

Default lesson behavior:
- supervised paired training
- default scale `x2`
- 1-epoch smoke runs must work on CPU

Recommended outputs:
- `config.json`
- `metrics.jsonl`
- `checkpoints/checkpoint.pt`
- `predictions.pt`
- optional `preview.png` if `torchvision` is available

Recommended metrics:
- `l1_loss`
- `psnr`

---

## Error Handling

Zoo-level behavior:
- unknown prefix -> `ValueError` with a hint to run `scripts/super_resolution_zoo.py --list`
- unknown arch -> `UnknownLocalArch` with the same hint

Builder-level behavior:
- invalid variant -> `ValueError`
- `in_channels <= 0` -> `ValueError`
- `upscale_factor < 2` -> `ValueError`
- unsupported factor in v1 -> `ValueError`

Model-level behavior:
- input must be 4D `(B, C, H, W)`
- spatial dimensions must be large enough for the requested degradation / upsampling path

---

## Testing

Add focused tests for the zoo:
- listing arches
- building representative families
- forward smoke with output shape checks
- CLI `--list`
- CLI `--smoke`

Add focused tests for the lesson:
- synthetic `(lr, hr)` batch generation smoke
- one-step training smoke
- output artifact checks

Initial expected zoo assertions:
- at least 18 arches
- `sr:srcnn_tiny`
- `sr:fsrcnn_small`
- `sr:edsr_sr_base`
- `sr:rcan_sr_tiny`
- `sr:rdn_sr_small`
- `sr:swinir_sr_tiny`

---

## Non-Goals for v1

Do not include in the first version:
- GAN-based SR training
- perceptual loss or VGG-feature losses
- realistic blind degradation modeling
- multi-scale curriculum
- external datasets or downloads

The first version should optimize for:
- clear API shape
- CPU smoke stability
- local educational value
- easy future extension
