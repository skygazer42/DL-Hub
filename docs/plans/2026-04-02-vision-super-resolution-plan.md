# Vision Super-Resolution Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a local toy-first image super-resolution zoo under `dlhub.vision`, plus a CPU-friendly paired synthetic training lesson and focused smoke tests.

**Architecture:** Create a dedicated `dlhub/vision/super_resolution/` package with one family per file, each exposing `_VARIANTS` and `build_*_super_resolver(...)`. Reuse the repository's AST-based lazy zoo discovery pattern and follow `@test-driven-development` for each task before touching production code.

**Tech Stack:** Python, PyTorch, AST-based lazy registry discovery, argparse, pytest

---

### Task 1: Add the failing zoo and lesson tests

**Files:**
- Create: `F:/DL-Hub/tests/test_dlhub_vision_super_resolution_zoo.py`
- Create: `F:/DL-Hub/tests/test_tracks_vision_super_resolution.py`

**Step 1: Write the failing zoo test**

Add tests that:
- expect at least 18 arches
- expect `sr:srcnn_tiny`, `sr:fsrcnn_small`, `sr:edsr_sr_base`, `sr:rcan_sr_tiny`, `sr:rdn_sr_small`, `sr:swinir_sr_tiny`
- build representative families and check that the model returns a dict containing `sr`
- assert that `sr` has shape `(B, C, H * 2, W * 2)` for `x2` super-resolution
- run CLI `--list` and `--smoke`

**Step 2: Write the failing lesson smoke test**

Add tests that:
- build one `(lr, hr)` batch and assert the paired spatial-size relationship
- run a one-epoch or one-batch training smoke
- verify `config.json`, `metrics.jsonl`, and `checkpoints/checkpoint.pt`

**Step 3: Run tests to verify they fail**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q`
- `python -m pytest tests/test_tracks_vision_super_resolution.py -q`

Expected:
- both fail because the module, script, and lesson do not exist yet

**Step 4: Commit**

```bash
git add tests/test_dlhub_vision_super_resolution_zoo.py tests/test_tracks_vision_super_resolution.py
git commit -m "test: add failing super-resolution coverage"
```

### Task 2: Add shared super-resolution utilities and package wiring

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/_common.py`

**Step 1: Write the failing package-level expectation**

Extend `tests/test_dlhub_vision_super_resolution_zoo.py` if needed so the first failure after import creation is about missing builders or missing valid output, not missing modules.

**Step 2: Run the focused test to keep it red**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q`

Expected:
- failure moves from missing package to missing families or zoo logic

**Step 3: Write minimal shared utilities**

Add helpers for:
- validating `(B, C, H, W)` SR input
- validating `upscale_factor == 2`
- default variant generation helper
- pixel-shuffle upsampling block
- residual block for SR families
- small channel-attention helper
- PSNR helper or lightweight metric helper if needed by the lesson

**Step 4: Re-run the focused test**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q`

Expected:
- still red, now failing because family modules and zoo are absent

**Step 5: Commit**

```bash
git add dlhub/vision/super_resolution/__init__.py dlhub/vision/super_resolution/_common.py tests/test_dlhub_vision_super_resolution_zoo.py
git commit -m "feat: add super-resolution shared utilities"
```

### Task 3: Implement the first two CNN families

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/srcnn.py`
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/fsrcnn.py`

**Step 1: Narrow the failing test to these families**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q -k "srcnn or fsrcnn"`

Expected:
- fail because builders are not available yet

**Step 2: Write minimal implementations**

Implement:
- `build_srcnn_super_resolver(...)`
- `build_fsrcnn_super_resolver(...)`

Requirements:
- define `_VARIANTS`
- accept `in_channels`, `variant`, `upscale_factor`
- reject unsupported factors
- return a dict with `sr`

**Step 3: Re-run the focused test**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q -k "srcnn or fsrcnn"`

Expected:
- `srcnn` and `fsrcnn` cases pass
- other SR families still fail

**Step 4: Commit**

```bash
git add dlhub/vision/super_resolution/srcnn.py dlhub/vision/super_resolution/fsrcnn.py
git commit -m "feat: add srcnn and fsrcnn super-resolution families"
```

### Task 4: Implement the residual SR families

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/edsr_sr.py`
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/rcan_sr.py`
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/rdn_sr.py`

**Step 1: Run the focused residual-family tests**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q -k "edsr_sr or rcan_sr or rdn_sr"`

Expected:
- fail because these families are not implemented yet

**Step 2: Write minimal implementations**

Implement:
- `build_edsr_sr_super_resolver(...)`
- `build_rcan_sr_super_resolver(...)`
- `build_rdn_sr_super_resolver(...)`

Requirements:
- lightweight toy-first versions only
- shared residual building blocks from `_common.py`
- output dict must always include `sr`

**Step 3: Re-run the focused residual-family tests**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q -k "edsr_sr or rcan_sr or rdn_sr"`

Expected:
- residual-family cases pass
- transformer family and zoo/CLI cases still fail

**Step 4: Commit**

```bash
git add dlhub/vision/super_resolution/edsr_sr.py dlhub/vision/super_resolution/rcan_sr.py dlhub/vision/super_resolution/rdn_sr.py
git commit -m "feat: add residual super-resolution families"
```

### Task 5: Implement the lightweight transformer SR family

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/super_resolution/swinir_sr.py`

**Step 1: Run the focused transformer-family test**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q -k "swinir_sr"`

Expected:
- fail because the family is missing

**Step 2: Write the minimal implementation**

Implement:
- `build_swinir_sr_super_resolver(...)`

Requirements:
- use a small toy windowed-attention or transformer-inspired block
- keep CPU smoke cost small
- output dict must include `sr`

**Step 3: Re-run the focused transformer-family test**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q -k "swinir_sr"`

Expected:
- transformer-family case passes
- zoo/CLI and lesson tests still fail

**Step 4: Commit**

```bash
git add dlhub/vision/super_resolution/swinir_sr.py
git commit -m "feat: add lightweight swinir-style super-resolution family"
```

### Task 6: Add the zoo registry and CLI

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/super_resolution_zoo.py`
- Create: `F:/DL-Hub/scripts/super_resolution_zoo.py`

**Step 1: Run the zoo test to confirm remaining failures**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q`

Expected:
- failure now centers on missing zoo registry and script behavior

**Step 2: Implement the lazy registry**

Mirror the structure of:
- `F:/DL-Hub/dlhub/vision/style_transfer_zoo.py`
- `F:/DL-Hub/dlhub/vision/video_summarization_zoo.py`

Requirements:
- prefix `sr:`
- AST discovery of `_VARIANTS`
- lazy import of the family module
- clear `UnknownLocalArch` error for unknown names

**Step 3: Implement the CLI**

Support:
- `--list`
- `--search`
- `--limit`
- `--smoke`
- `--batch-size`
- `--image-size`
- `--in-channels`
- `--upscale-factor`
- `--width-mult`
- `--dropout`

**Step 4: Re-run the zoo test**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q`

Expected:
- zoo test passes

**Step 5: Commit**

```bash
git add dlhub/vision/super_resolution_zoo.py scripts/super_resolution_zoo.py tests/test_dlhub_vision_super_resolution_zoo.py
git commit -m "feat: add super-resolution zoo and cli"
```

### Task 7: Add the synthetic paired-supervision lesson data and model glue

**Files:**
- Create: `F:/DL-Hub/tracks/vision/lesson_17_synthetic_super_resolution/__init__.py`
- Create: `F:/DL-Hub/tracks/vision/lesson_17_synthetic_super_resolution/data.py`
- Create: `F:/DL-Hub/tracks/vision/lesson_17_synthetic_super_resolution/model.py`

**Step 1: Run the lesson test to confirm the current failure**

Run:
- `python -m pytest tests/test_tracks_vision_super_resolution.py -q`

Expected:
- failure because lesson package and data/model code do not exist

**Step 2: Write minimal dataset and model glue**

Implement:
- HR synthetic image generation
- paired degradation pipeline producing `(lr, hr)`
- dataloader helpers or batch helpers
- model builder wrapper that delegates to `dlhub.vision.super_resolution_zoo.build_local_model`

Requirements:
- default paired `x2` setup
- no downloads
- deterministic behavior under seed control

**Step 3: Re-run the lesson test**

Run:
- `python -m pytest tests/test_tracks_vision_super_resolution.py -q`

Expected:
- failure moves from missing lesson package to missing training entrypoint

**Step 4: Commit**

```bash
git add tracks/vision/lesson_17_synthetic_super_resolution/__init__.py tracks/vision/lesson_17_synthetic_super_resolution/data.py tracks/vision/lesson_17_synthetic_super_resolution/model.py tests/test_tracks_vision_super_resolution.py
git commit -m "feat: add synthetic super-resolution lesson data pipeline"
```

### Task 8: Add the lesson training entrypoint and lesson README

**Files:**
- Create: `F:/DL-Hub/tracks/vision/lesson_17_synthetic_super_resolution/train.py`
- Create: `F:/DL-Hub/tracks/vision/lesson_17_synthetic_super_resolution/README.md`

**Step 1: Run the lesson smoke test again**

Run:
- `python -m pytest tests/test_tracks_vision_super_resolution.py -q`

Expected:
- fail because training smoke cannot run end-to-end yet

**Step 2: Write the minimal training loop**

Implement:
- argument parsing
- run/data config dataclasses
- output directory creation via `dlhub.paths.build_run_paths`
- one-epoch supervised train/eval loop
- metric logging to `metrics.jsonl`
- checkpoint save to `checkpoints/checkpoint.pt`
- prediction artifact save to `predictions.pt`
- optional preview save if `torchvision` is available

**Step 3: Add the lesson README**

Document:
- what the lesson trains
- example run command
- output directory structure
- recommended starter arches: `sr:srcnn_tiny`, `sr:edsr_sr_tiny`

**Step 4: Re-run the lesson smoke test**

Run:
- `python -m pytest tests/test_tracks_vision_super_resolution.py -q`

Expected:
- lesson smoke test passes

**Step 5: Commit**

```bash
git add tracks/vision/lesson_17_synthetic_super_resolution/train.py tracks/vision/lesson_17_synthetic_super_resolution/README.md tests/test_tracks_vision_super_resolution.py
git commit -m "feat: add synthetic super-resolution lesson"
```

### Task 9: Update track docs and run final verification

**Files:**
- Modify: `F:/DL-Hub/tracks/vision/README.md`
- Modify: `F:/DL-Hub/docs/plans/2026-04-02-vision-super-resolution-design.md`

**Step 1: Add the new lesson entry**

Update the Vision track README to include:
- `lesson_17_synthetic_super_resolution/`

Keep the style aligned with the existing lesson list.

**Step 2: Reconcile the design doc if names changed**

If any implementation detail differs from the approved design, update the design doc so the record matches what shipped.

**Step 3: Run the focused verification suite**

Run:
- `python -m pytest tests/test_dlhub_vision_super_resolution_zoo.py -q`
- `python -m pytest tests/test_tracks_vision_super_resolution.py -q`
- `python scripts/super_resolution_zoo.py --list --limit 8`
- `python scripts/super_resolution_zoo.py --smoke sr:srcnn_tiny`

Expected:
- both test files pass
- CLI list prints the SR zoo header and total count
- CLI smoke prints the requested `sr:srcnn_tiny` run summary

**Step 4: Perform completion verification**

Use `@verification-before-completion` and verify:
- fresh commands were run
- outputs match the claimed status
- no missing artifacts remain

**Step 5: Commit**

```bash
git add tracks/vision/README.md docs/plans/2026-04-02-vision-super-resolution-design.md
git commit -m "docs: add super-resolution lesson to vision track"
```
