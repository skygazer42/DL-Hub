# Multimodal Lesson 15 2D-TAN-Lite Temporal Grounding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding` as a teaching lesson for text-conditioned temporal localization with a dense 2D segment map.

**Architecture:** The lesson will reuse the short-video single-query temporal grounding setting from lesson 14, but replace boundary and proposal matching heads with a dense `T x T` temporal map. A small frame encoder and temporal encoder will produce per-frame features, a small text encoder will condition them on the query, and a compact 2D convolutional head will score every valid segment cell directly.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_2dtan.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_15_2dtan_toy_temporal_grounding` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_2dtan.py`

Expected: FAIL because lesson 15 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the dense temporal-map dataset

**Files:**
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/__init__.py`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/data.py`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/README.md`
- Test: `tests/test_tracks_multimodal_2dtan.py`

**Step 1: Write the failing test**

Require:

- `video` shaped `(B, T, 3, H, W)`
- `query_ids` and `attention_mask` shaped `(B, L)`
- `map_labels` and `map_mask` shaped `(B, T, T)`
- `segment` shaped `(B, 2)`
- vocabulary entries for `when`, `does`, `move`, `left`, `right`, and `flash`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_2dtan.py::test_multimodal_2dtan_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- short single-object event video renderer
- dense temporal-IoU map targets
- upper-triangular validity mask
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the 2D-TAN-lite model

**Files:**
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/model.py`
- Test: `tests/test_tracks_multimodal_2dtan.py`

**Step 1: Write the failing test**

Require:

- model outputs include `score_map`, `map_features`, and `pred_segments`
- shapes `(B, T, T)`, `(B, T, T, H)`, and `(B, 2)`
- finite masked temporal-map loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_2dtan.py::test_multimodal_2dtan_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- shared frame encoder
- lightweight temporal encoder
- lightweight text encoder
- dense segment-map construction from `(start, end, pooled, query)` features
- compact 2D convolutional scoring head
- decoding helpers and temporal grounding metrics

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and track integration

**Files:**
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_2dtan.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.train --epochs 1 --num-samples 64 --batch-size 8 --num-frames 8 --image-size 20 --max-text-length 16 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_2dtan_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_15_2dtan_toy_temporal_grounding/pytest_2dtan_smoke/config.json`
- `outputs/multimodal/lesson_15_2dtan_toy_temporal_grounding/pytest_2dtan_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_15_2dtan_toy_temporal_grounding/pytest_2dtan_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_2dtan.py::test_multimodal_2dtan_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train and eval loops
- sample logging with temporal IoU
- config, vocab, metrics, checkpoint writing
- track README update for lesson 15

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_2dtan.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/__init__.py`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/data.py`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/model.py`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/train.py`
- Create: `tracks/multimodal/lesson_15_2dtan_toy_temporal_grounding/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_2dtan.py tests/test_tracks_multimodal_bmn.py tests/test_tracks_multimodal_video_vlm.py tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-15-multimodal-2dtan-lesson-design.md docs/plans/2026-03-15-multimodal-2dtan-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_video_vlm.py tests/test_tracks_multimodal_bmn.py tests/test_tracks_multimodal_2dtan.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_15_2dtan_toy_temporal_grounding --dry-run`

Expected: lesson 15 appears in the listing and resolves to the train module.
