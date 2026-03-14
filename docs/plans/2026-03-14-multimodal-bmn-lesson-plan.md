# Multimodal Lesson 14 BMN-Lite Temporal Grounding Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding` as a teaching lesson for text-conditioned temporal localization over short videos.

**Architecture:** The lesson will synthesize short videos with a single object and one target event segment. A small frame encoder and temporal encoder will produce per-frame features, a small text encoder will condition them on the query, and three heads will predict start logits, end logits, and an upper-triangular BMN-lite proposal map.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_bmn.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_14_bmn_toy_temporal_grounding` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_bmn.py`

Expected: FAIL because lesson 14 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the temporal grounding dataset

**Files:**
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/__init__.py`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/data.py`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/README.md`
- Test: `tests/test_tracks_multimodal_bmn.py`

**Step 1: Write the failing test**

Require:

- `video` shaped `(B, T, 3, H, W)`
- `query_ids` and `attention_mask` shaped `(B, L)`
- `start_labels` shaped `(B, T)`
- `end_labels` shaped `(B, T)`
- `proposal_labels` and `proposal_mask` shaped `(B, T, T)`
- `segment` shaped `(B, 2)`
- vocabulary entries for `when`, `does`, `move`, `left`, `right`, and `flash`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_bmn.py::test_multimodal_bmn_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- single-object event video renderer
- boundary labels and proposal IoU targets
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the BMN-lite temporal grounding model

**Files:**
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/model.py`
- Test: `tests/test_tracks_multimodal_bmn.py`

**Step 1: Write the failing test**

Require:

- model outputs include `start_logits`, `end_logits`, `proposal_scores`, and `pred_segments`
- shapes `(B, T)`, `(B, T)`, `(B, T, T)`, and `(B, 2)`
- finite temporal grounding loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_bmn.py::test_multimodal_bmn_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- shared frame encoder
- lightweight temporal encoder
- lightweight text encoder
- query-conditioned temporal fusion
- start, end, and proposal heads
- temporal IoU based decoding helpers and metrics

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and track integration

**Files:**
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_bmn.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.train --epochs 1 --num-samples 64 --batch-size 8 --num-frames 8 --image-size 20 --max-text-length 16 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_bmn_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_14_bmn_toy_temporal_grounding/pytest_bmn_smoke/config.json`
- `outputs/multimodal/lesson_14_bmn_toy_temporal_grounding/pytest_bmn_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_14_bmn_toy_temporal_grounding/pytest_bmn_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_bmn.py::test_multimodal_bmn_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train and eval loops
- sample logging with temporal IoU
- config, vocab, metrics, checkpoint writing
- track README update for lesson 14

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_bmn.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/__init__.py`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/data.py`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/model.py`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/train.py`
- Create: `tracks/multimodal/lesson_14_bmn_toy_temporal_grounding/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_bmn.py tests/test_tracks_multimodal_video_vlm.py tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-bmn-lesson-design.md docs/plans/2026-03-14-multimodal-bmn-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_video_vlm.py tests/test_tracks_multimodal_bmn.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_14_bmn_toy_temporal_grounding --dry-run`

Expected: lesson 14 appears in the listing and resolves to the train module.
