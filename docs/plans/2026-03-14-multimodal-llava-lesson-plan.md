# Multimodal Lesson 03 LLaVA-Lite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_03_llava_toy_instruction_vlm` as a teaching lesson for single-turn visual question answering with short generated answers.

**Architecture:** The lesson will define its own synthetic instruction-VLM dataset, lesson-local vocabulary, tiny CNN visual token encoder, explicit vision projector, and a causal decoder-style language model that consumes projected visual tokens as a prefix before textual instruction and answer tokens. Training will supervise only the answer span with a decoder-only loss.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and API contracts with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_llava.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_03_llava_toy_instruction_vlm` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_03_llava_toy_instruction_vlm.train`
- batch dictionaries containing instruction VLM keys
- forward outputs containing `logits` and `visual_tokens`
- a tiny training smoke run writing standard artifacts

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_llava.py`

Expected: FAIL because lesson 3 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the first failing assertion.

**Step 4: Run test to verify it still fails for the next missing behavior**

Re-run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_llava.py`

Use the next failure as the next implementation target.

### Task 2: Build the synthetic instruction-VLM dataset

**Files:**
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/data.py`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/README.md`
- Test: `tests/test_tracks_multimodal_llava.py`

**Step 1: Write the failing test**

Add a test that asserts:

- image batch shape is `(B, 3, H, W)`
- `instruction_ids`, `input_ids`, `labels`, and `attention_mask` have consistent shapes
- `question_type` is present
- the vocabulary includes question and answer tokens such as `what`, `where`, `yes`, `no`, `top`, `left`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_llava.py::test_multimodal_llava_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocabulary with BOS, EOS, and separator
- templated instruction generation for five question families
- short answer construction
- labels that ignore non-answer positions
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the LLaVA-lite teaching model

**Files:**
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/model.py`
- Test: `tests/test_tracks_multimodal_llava.py`

**Step 1: Write the failing test**

Add tests that require:

- output logits shaped `(B, T, V)`
- projected visual tokens shaped `(B, N, H)`
- finite QA loss
- greedy generation to return short answers

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_llava.py::test_multimodal_llava_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- `VisionEncoder`
- `VisionProjector`
- `TinyMultimodalDecoderLM`
- `ToyLLaVAModel`
- QA loss and accuracy helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/train.py`
- Test: `tests/test_tracks_multimodal_llava.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_03_llava_toy_instruction_vlm.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 16 --max-text-length 12 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_llava_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_03_llava_toy_instruction_vlm/pytest_llava_smoke/config.json`
- `outputs/multimodal/lesson_03_llava_toy_instruction_vlm/pytest_llava_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_03_llava_toy_instruction_vlm/pytest_llava_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_llava.py::test_multimodal_llava_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI argument parsing
- instruction-VLM training loop
- metric computation for exact match and yes/no subset
- config, vocab, metrics, sample logging, and checkpoint writing

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_llava.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/data.py`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/model.py`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/train.py`
- Create: `tracks/multimodal/lesson_03_llava_toy_instruction_vlm/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_03_llava_toy_instruction_vlm --dry-run`

Expected: lesson 3 appears in the track listing and resolves to the train module.
