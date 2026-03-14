# Multimodal Lesson 07 Q-Former-Lite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm` as a teaching lesson for Q-Former-style visual bottleneck bridging.

**Architecture:** The lesson will generate single-image QA examples over synthetic objects. A small CNN will emit spatial visual tokens, a learnable query bottleneck will compress them into a fixed number of query states through cross-attention, and a tiny decoder LM will answer the question from those query states.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock discovery and lesson contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_qformer.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_07_qformer_toy_bridge_vlm` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_07_qformer_toy_bridge_vlm.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_qformer.py`

Expected: FAIL because lesson 7 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the single-image QA dataset

**Files:**
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/data.py`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/README.md`
- Test: `tests/test_tracks_multimodal_qformer.py`

**Step 1: Write the failing test**

Require:

- `image` shaped `(B, 3, H, W)`
- `question_ids`, `input_ids`, `labels`, `attention_mask` with consistent `(B, T)` shapes
- vocab entries for `what`, `where`, `yes`, `no`, `color`, `shape`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_qformer.py::test_multimodal_qformer_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- synthetic single-object QA data
- question/answer formatting
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the Q-Former-lite model

**Files:**
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/model.py`
- Test: `tests/test_tracks_multimodal_qformer.py`

**Step 1: Write the failing test**

Require:

- model outputs include `logits` and `query_states`
- logits shaped `(B, T, vocab_size)`
- query states shaped `(B, num_query_tokens, hidden_dim)`
- finite QA loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_qformer.py::test_multimodal_qformer_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- tiny vision encoder that emits visual tokens
- learned query-token bridge with lightweight cross-attention
- decoder-style LM fed by query states
- QA loss and metric helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_qformer.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_07_qformer_toy_bridge_vlm.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 16 --max-text-length 12 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_qformer_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_07_qformer_toy_bridge_vlm/pytest_qformer_smoke/config.json`
- `outputs/multimodal/lesson_07_qformer_toy_bridge_vlm/pytest_qformer_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_07_qformer_toy_bridge_vlm/pytest_qformer_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_qformer.py::test_multimodal_qformer_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train/eval loops
- sample logging
- config, vocab, metrics, checkpoint writing
- track README update for lesson 7

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_qformer.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/data.py`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/model.py`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/train.py`
- Create: `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_qformer.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-qformer-lesson-design.md docs/plans/2026-03-14-multimodal-qformer-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_qformer.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_07_qformer_toy_bridge_vlm --dry-run`

Expected: lesson 7 appears in the listing and resolves to the train module.
