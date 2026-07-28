# Multimodal Lesson 12 Key-Value OCR-Lite Implementation Plan

**Goal:** Add `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm` as a teaching lesson for prompt-conditioned key-value OCR on synthetic document images.

**Architecture:** The lesson will synthesize small document images with several `key: value` rows and a prompt such as `read total`. A tiny CNN will encode the document image, visual tokens will be prefixed into a decoder-style LM, and the model will generate the requested value or `none` if the field is absent.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_key_value_ocr.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_12_key_value_ocr_compact_doc_vlm` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_12_key_value_ocr_compact_doc_vlm.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_key_value_ocr.py`

Expected: FAIL because lesson 12 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the key-value OCR dataset

**Files:**
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/data.py`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/README.md`
- Test: `tests/test_tracks_multimodal_key_value_ocr.py`

**Step 1: Write the failing test**

Require:

- `image` shaped `(B, 3, H, W)`
- `prompt_ids`, `input_ids`, `labels`, `attention_mask` with shape `(B, T)`
- `present` shaped `(B,)`
- vocabulary entries for `read`, `name`, `total`, `none`, and at least one value token

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_key_value_ocr.py::test_multimodal_key_value_ocr_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab with prompt and answer tokens
- synthetic document renderer with several `key: value` rows
- positive and negative field sampling
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the document OCR model

**Files:**
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/model.py`
- Test: `tests/test_tracks_multimodal_key_value_ocr.py`

**Step 1: Write the failing test**

Require:

- model outputs include `logits` and `visual_tokens`
- logits shaped `(B, T, vocab_size)`
- finite OCR loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_key_value_ocr.py::test_multimodal_key_value_ocr_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- tiny document vision encoder
- decoder-style LM with visual prefix
- generation helpers and present-accuracy metrics

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and track integration

**Files:**
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_key_value_ocr.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_12_key_value_ocr_compact_doc_vlm.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 32 --max-text-length 20 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_key_value_ocr_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/pytest_key_value_ocr_smoke/config.json`
- `outputs/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/pytest_key_value_ocr_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/pytest_key_value_ocr_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_key_value_ocr.py::test_multimodal_key_value_ocr_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train and eval loops
- sample logging
- config, vocab, metrics, checkpoint writing
- track README update for lesson 12

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_key_value_ocr.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/data.py`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/model.py`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/train.py`
- Create: `tracks/multimodal/lesson_12_key_value_ocr_compact_doc_vlm/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-key-value-ocr-lesson-design.md docs/plans/2026-03-14-multimodal-key-value-ocr-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_key_value_ocr.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_12_key_value_ocr_compact_doc_vlm --dry-run`

Expected: lesson 12 appears in the listing and resolves to the train module.
