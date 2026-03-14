# Multimodal VLM Lessons Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new `tracks/multimodal` teaching track and implement `lesson_01_clip_toy_retrieval` as an independent educational CLIP-style model with synthetic data, a runnable training script, and focused tests.

**Architecture:** The new track will mirror existing lesson conventions in `tracks/vision`, `tracks/llm`, and `tracks/generative`. Lesson 1 will define its own synthetic dataset, text vocabulary, CNN and text encoders, projection heads, and symmetric contrastive training loop, without importing zoo family implementations. The script will plug into the existing `scripts/run_lesson.py` discovery flow.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock the new track and lesson contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_clip.py`

**Step 1: Write the failing test**

Add tests that require:

- `multimodal` to appear in `python scripts/run_lesson.py --list`
- `lesson_01_clip_toy_retrieval` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_01_clip_toy_retrieval.train`
- a teaching model API that returns `image_embed`, `text_embed`, `logits_per_image`, and `logits_per_text`
- a tiny training run to produce standard output artifacts

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py`

Expected: FAIL because `tracks/multimodal` and the lesson package do not exist yet.

**Step 3: Write minimal implementation**

Create only the package skeleton and the minimum code needed to satisfy the first failing assertion.

**Step 4: Run test to verify it still fails for the next missing behavior**

Re-run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py`

Use the next failure as the next implementation target.

### Task 2: Build the synthetic multimodal dataset

**Files:**
- Create: `tracks/multimodal/__init__.py`
- Create: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/__init__.py`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/data.py`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/README.md`
- Test: `tests/test_tracks_multimodal_clip.py`

**Step 1: Write the failing test**

Add a test that constructs a tiny dataset and dataloader, then asserts:

- image tensor shape is `(B, 3, H, W)`
- token ids and masks have consistent shapes
- vocabulary contains attribute tokens used in captions

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_clip.py::test_multimodal_clip_batch_shapes`

Expected: FAIL because `data.py` and the lesson package are missing.

**Step 3: Write minimal implementation**

Implement:

- a small lesson-local vocabulary
- deterministic synthetic attribute combinations
- tensor image rendering in pure PyTorch
- a collate function and dataloader helper

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the independent teaching model

**Files:**
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/model.py`
- Test: `tests/test_tracks_multimodal_clip.py`

**Step 1: Write the failing test**

Add tests that require:

- the forward pass to return the expected keys
- shared embedding dimensions to match config
- similarity logits to be square over the batch
- the contrastive loss helper to return a finite scalar

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_clip.py::test_multimodal_clip_model_outputs`

Expected: FAIL because `model.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- `ModelConfig`
- `VisionEncoder`
- `TextEncoder`
- `ProjectionHead`
- `ToyCLIPModel`
- `clip_contrastive_loss`

Keep the code explicit, small, and lesson-first.

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and run smoke

**Files:**
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/train.py`
- Test: `tests/test_tracks_multimodal_clip.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_01_clip_toy_retrieval.train --epochs 1 --num-samples 64 --batch-size 8 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_clip`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_01_clip_toy_retrieval/pytest_clip/config.json`
- `outputs/multimodal/lesson_01_clip_toy_retrieval/pytest_clip/metrics.jsonl`
- `outputs/multimodal/lesson_01_clip_toy_retrieval/pytest_clip/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_clip.py::test_multimodal_clip_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- dataloader/model setup
- symmetric contrastive train and eval loops
- config, metrics, and checkpoint writing

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_clip.py`
- Create: `tracks/multimodal/__init__.py`
- Create: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/__init__.py`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/data.py`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/model.py`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/train.py`
- Create: `tracks/multimodal/lesson_01_clip_toy_retrieval/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py --list`
- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_01_clip_toy_retrieval --dry-run`

Expected: `multimodal` is discoverable and the lesson resolves to the train module.
