# Multimodal Lesson 11 Grounded-SAM-Lite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation` as a teaching lesson for open-vocabulary text-conditioned segmentation.

**Architecture:** The lesson will generate multi-object synthetic scenes with unique color-shape categories. A text query may or may not match an object in the image. A tiny CNN plus text prompt encoder will drive a lightweight mask decoder that predicts presence and a low-resolution target mask.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_grounded_sam.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_11_grounded_sam_toy_open_vocab_segmentation` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_grounded_sam.py`

Expected: FAIL because lesson 11 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the open-vocabulary segmentation dataset

**Files:**
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/__init__.py`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/data.py`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/README.md`
- Test: `tests/test_tracks_multimodal_grounded_sam.py`

**Step 1: Write the failing test**

Require:

- `image` shaped `(B, 3, H, W)`
- `query_ids`, `attention_mask` with shape `(B, T)`
- `target_present` shaped `(B,)`
- `target_mask` shaped `(B, 1, M, M)`
- vocab entries for `segment`, `mask`, `red`, `square`, `circle`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_grounded_sam.py::test_multimodal_grounded_sam_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- multi-object scene generator with unique query categories
- positive and negative query sampling
- low-resolution target mask generation
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the Grounded-SAM-lite model

**Files:**
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/model.py`
- Test: `tests/test_tracks_multimodal_grounded_sam.py`

**Step 1: Write the failing test**

Require:

- model outputs include `presence_logit`, `mask_logits`, and `pred_mask`
- presence logits shaped `(B,)`
- mask logits shaped `(B, 1, mask_size, mask_size)`
- finite segmentation loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_grounded_sam.py::test_multimodal_grounded_sam_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- tiny image encoder
- text prompt encoder
- lightweight mask decoder
- presence head
- masked segmentation loss and metrics

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and track integration

**Files:**
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_grounded_sam.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 32 --mask-size 8 --max-text-length 6 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_grounded_sam_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/pytest_grounded_sam_smoke/config.json`
- `outputs/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/pytest_grounded_sam_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/pytest_grounded_sam_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_grounded_sam.py::test_multimodal_grounded_sam_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train/eval loops
- sample logging
- config, vocab, metrics, checkpoint writing
- track README update for lesson 11

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_grounded_sam.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/__init__.py`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/data.py`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/model.py`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/train.py`
- Create: `tracks/multimodal/lesson_11_grounded_sam_toy_open_vocab_segmentation/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-grounded-sam-lesson-design.md docs/plans/2026-03-14-multimodal-grounded-sam-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_grounded_sam.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_11_grounded_sam_toy_open_vocab_segmentation --dry-run`

Expected: lesson 11 appears in the listing and resolves to the train module.
