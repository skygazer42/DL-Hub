# Multimodal Lesson 02 BLIP-Lite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `tracks/multimodal/lesson_02_blip_toy_captioning` as a BLIP-lite teaching lesson with synthetic caption generation, image-text matching, and a runnable training script.

**Architecture:** The lesson will define its own synthetic multimodal dataset, lesson-local vocabulary, tiny CNN visual token encoder, decoder with visual cross-attention, and an ITM classification head. The same fused text path will support both caption generation and image-text matching. The implementation will remain independent from `dlhub.multimodal.vlm` zoo files.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock lesson discovery and API contracts with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_blip.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_02_blip_toy_captioning` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_02_blip_toy_captioning.train`
- batch dictionaries containing captioning and ITM keys
- forward outputs containing caption logits and ITM logits
- a tiny training smoke run to write standard artifacts

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_blip.py`

Expected: FAIL because lesson 2 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed to satisfy the first failing assertion.

**Step 4: Run test to verify it still fails for the next missing behavior**

Re-run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_blip.py`

Use the next failure as the next implementation target.

### Task 2: Build the synthetic captioning + ITM dataset

**Files:**
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/__init__.py`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/data.py`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/README.md`
- Test: `tests/test_tracks_multimodal_blip.py`

**Step 1: Write the failing test**

Add a test that asserts:

- image batch shape is `(B, 3, H, W)`
- caption decoder inputs and outputs share the same shape
- ITM tokens and labels are present
- vocabulary contains sentence tokens such as `a`, `at`, and attribute words

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_blip.py::test_multimodal_blip_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocabulary with BOS and EOS
- templated sentence captions
- hard-negative caption generation for ITM
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the BLIP-lite teaching model

**Files:**
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/model.py`
- Test: `tests/test_tracks_multimodal_blip.py`

**Step 1: Write the failing test**

Add tests that require:

- caption logits shaped `(B, T, V)`
- ITM logits shaped `(B, 2)`
- optional fused hidden states for inspection
- finite combined loss from caption and ITM outputs

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_blip.py::test_multimodal_blip_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- `ModelConfig`
- `VisionEncoder`
- additive cross-attention module
- decoder with teacher forcing
- `ITMHead`
- combined loss helper

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/train.py`
- Test: `tests/test_tracks_multimodal_blip.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_02_blip_toy_captioning.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 16 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_blip_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_02_blip_toy_captioning/pytest_blip_smoke/config.json`
- `outputs/multimodal/lesson_02_blip_toy_captioning/pytest_blip_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_02_blip_toy_captioning/pytest_blip_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_blip.py::test_multimodal_blip_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI argument parsing
- combined generation + ITM training loop
- metric computation
- config, vocab, metrics, sample logging, and checkpoint writing

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_blip.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/__init__.py`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/data.py`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/model.py`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/train.py`
- Create: `tracks/multimodal/lesson_02_blip_toy_captioning/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_blip.py tests/test_scripts_run_lesson.py`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_02_blip_toy_captioning --dry-run`

Expected: lesson 2 appears in the track listing and resolves to the train module.
