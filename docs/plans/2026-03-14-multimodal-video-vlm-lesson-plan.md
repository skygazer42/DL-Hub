# Multimodal Lesson 13 Video-VLM-Lite Implementation Plan

**Goal:** Add `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa` as a teaching lesson for short-video temporal QA.

**Architecture:** The lesson will synthesize short compact videos with one moving colored shape. A shared frame encoder will process every frame, a lightweight temporal aggregator will combine frame features into video tokens, and a decoder-style LM will answer prompt-conditioned questions about color, shape, and motion direction.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_video_vlm.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_13_video_vlm_compact_temporal_qa` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_13_video_vlm_compact_temporal_qa.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_video_vlm.py`

Expected: FAIL because lesson 13 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the temporal QA dataset

**Files:**
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/__init__.py`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/data.py`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/README.md`
- Test: `tests/test_tracks_multimodal_video_vlm.py`

**Step 1: Write the failing test**

Require:

- `video` shaped `(B, T, 3, H, W)`
- `prompt_ids`, `input_ids`, `labels`, `attention_mask` with shape `(B, L)`
- `task_type` list length `B`
- vocabulary entries for `what`, `color`, `shape`, `moving`, `left`, `yes`, and `no`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_video_vlm.py::test_multimodal_video_vlm_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- single-object video renderer
- prompt and answer sampling
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the temporal video VLM model

**Files:**
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/model.py`
- Test: `tests/test_tracks_multimodal_video_vlm.py`

**Step 1: Write the failing test**

Require:

- model outputs include `logits` and `video_tokens`
- logits shaped `(B, L, vocab_size)`
- `video_tokens` with batch dimension `B` and hidden dimension `hidden_dim`
- finite QA loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_video_vlm.py::test_multimodal_video_vlm_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- shared frame encoder
- lightweight temporal aggregator
- decoder-style LM with video token prefix
- generation helpers and yes/no metric support

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and track integration

**Files:**
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_video_vlm.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_13_video_vlm_compact_temporal_qa.train --epochs 1 --num-samples 64 --batch-size 8 --seq-len 4 --image-size 20 --max-text-length 16 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_video_vlm_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_13_video_vlm_compact_temporal_qa/pytest_video_vlm_smoke/config.json`
- `outputs/multimodal/lesson_13_video_vlm_compact_temporal_qa/pytest_video_vlm_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_13_video_vlm_compact_temporal_qa/pytest_video_vlm_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_video_vlm.py::test_multimodal_video_vlm_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train and eval loops
- sample logging
- config, vocab, metrics, checkpoint writing
- track README update for lesson 13

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_video_vlm.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/__init__.py`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/data.py`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/model.py`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/train.py`
- Create: `tracks/multimodal/lesson_13_video_vlm_compact_temporal_qa/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_video_vlm.py tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-video-vlm-lesson-design.md docs/plans/2026-03-14-multimodal-video-vlm-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_perceiver.py tests/test_tracks_multimodal_paligemma.py tests/test_tracks_multimodal_owlvit.py tests/test_tracks_multimodal_grounded_sam.py tests/test_tracks_multimodal_key_value_ocr.py tests/test_tracks_multimodal_video_vlm.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_13_video_vlm_compact_temporal_qa --dry-run`

Expected: lesson 13 appears in the listing and resolves to the train module.
