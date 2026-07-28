# Multimodal Lesson 10 OWL-ViT-Lite Implementation Plan

**Goal:** Add `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection` as a teaching lesson for text-conditioned open-vocabulary detection.

**Architecture:** The lesson will generate multi-object synthetic scenes with unique color-shape categories. A text query may or may not match an object in the image. A tiny CNN plus text encoder will fuse query and image features to predict presence, the target cell, and box deltas.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock discovery and lesson contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_owlvit.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_10_owlvit_compact_open_vocab_detection` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_10_owlvit_compact_open_vocab_detection.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_owlvit.py`

Expected: FAIL because lesson 10 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the open-vocab detection dataset

**Files:**
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/__init__.py`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/data.py`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/README.md`
- Test: `tests/test_tracks_multimodal_owlvit.py`

**Step 1: Write the failing test**

Require:

- `image` shaped `(B, 3, H, W)`
- `query_ids`, `attention_mask` with shape `(B, T)`
- `target_present`, `target_cell`, `target_box`, `target_delta`
- vocab entries for `detect`, `find`, `red`, `square`, `circle`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_owlvit.py::test_multimodal_owlvit_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- multi-object scene generator with unique query categories
- positive and negative query sampling
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the OWL-ViT-lite model

**Files:**
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/model.py`
- Test: `tests/test_tracks_multimodal_owlvit.py`

**Step 1: Write the failing test**

Require:

- model outputs include `presence_logit`, `cell_logits`, and `pred_boxes`
- presence logits shaped `(B,)`
- cell logits shaped `(B, grid_size * grid_size)`
- pred boxes shaped `(B, 4)`
- finite detection loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_owlvit.py::test_multimodal_owlvit_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- tiny vision encoder
- text encoder
- per-cell query-conditioned fusion
- presence head, cell head, and box head
- masked detection loss and metrics

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_owlvit.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_10_owlvit_compact_open_vocab_detection.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 32 --grid-size 4 --max-text-length 6 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_owlvit_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_10_owlvit_compact_open_vocab_detection/pytest_owlvit_smoke/config.json`
- `outputs/multimodal/lesson_10_owlvit_compact_open_vocab_detection/pytest_owlvit_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_10_owlvit_compact_open_vocab_detection/pytest_owlvit_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_owlvit.py::test_multimodal_owlvit_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train/eval loops
- sample logging
- config, vocab, metrics, checkpoint writing
- track README update for lesson 10

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_owlvit.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/__init__.py`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/data.py`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/model.py`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/train.py`
- Create: `tracks/multimodal/lesson_10_owlvit_compact_open_vocab_detection/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-owlvit-lesson-design.md docs/plans/2026-03-14-multimodal-owlvit-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_owlvit.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_10_owlvit_compact_open_vocab_detection --dry-run`

Expected: lesson 10 appears in the listing and resolves to the train module.
