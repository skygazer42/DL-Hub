# Multimodal Lesson 05 Mask-Grounding-Lite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_05_mask_grounding_toy_refexp` as a teaching lesson for text-conditioned region grounding with low-resolution target masks.

**Architecture:** The lesson will define its own synthetic multi-object referring-expression dataset, lesson-local vocabulary, spatial CNN backbone, lightweight text encoder, per-location multimodal fusion, and a mask head that predicts a low-resolution binary mask. Training will combine BCE-with-logits and dice loss, then evaluate IoU, dice, and foreground accuracy.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and API contracts with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_mask_grounding.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_05_mask_grounding_toy_refexp` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_05_mask_grounding_toy_refexp.train`
- batch dictionaries containing mask-grounding fields
- forward outputs containing `mask_logits` and `pred_mask`
- a tiny training smoke run writing standard artifacts

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_mask_grounding.py`

Expected: FAIL because lesson 5 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the first failing assertion.

**Step 4: Run test to verify it still fails for the next missing behavior**

Re-run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_mask_grounding.py`

Use the next failure as the next implementation target.

### Task 2: Build the synthetic mask-grounding dataset

**Files:**
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/__init__.py`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/data.py`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/README.md`
- Test: `tests/test_tracks_multimodal_mask_grounding.py`

**Step 1: Write the failing test**

Add a test that asserts:

- image batch shape is `(B, 3, H, W)`
- `input_ids` and `attention_mask` have consistent shapes
- `target_mask` is shaped `(B, 1, Hm, Wm)`
- the vocabulary includes words such as `segment`, `mask`, `highlight`, `top`, `left`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_mask_grounding.py::test_multimodal_mask_grounding_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocabulary
- synthetic multi-object scene generation
- unique referring-expression construction
- full-resolution target shape mask
- downsampled supervision mask
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the mask-grounding model

**Files:**
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/model.py`
- Test: `tests/test_tracks_multimodal_mask_grounding.py`

**Step 1: Write the failing test**

Add tests that require:

- `mask_logits` shaped `(B, 1, Hm, Wm)`
- `pred_mask` shaped `(B, 1, Hm, Wm)`
- finite mask-grounding loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_mask_grounding.py::test_multimodal_mask_grounding_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- `VisionEncoder`
- `TextEncoder`
- spatial fusion
- `MaskGroundingHead`
- BCE + dice loss helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/train.py`
- Test: `tests/test_tracks_multimodal_mask_grounding.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_05_mask_grounding_toy_refexp.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 32 --mask-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_mask_grounding_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_05_mask_grounding_toy_refexp/pytest_mask_grounding_smoke/config.json`
- `outputs/multimodal/lesson_05_mask_grounding_toy_refexp/pytest_mask_grounding_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_05_mask_grounding_toy_refexp/pytest_mask_grounding_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_mask_grounding.py::test_multimodal_mask_grounding_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI argument parsing
- mask-grounding training loop
- metric computation for IoU, dice, and foreground accuracy
- config, vocab, metrics, sample logging, and checkpoint writing

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_mask_grounding.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/__init__.py`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/data.py`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/model.py`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/train.py`
- Create: `tracks/multimodal/lesson_05_mask_grounding_toy_refexp/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_05_mask_grounding_toy_refexp --dry-run`

Expected: lesson 5 appears in the track listing and resolves to the train module.
