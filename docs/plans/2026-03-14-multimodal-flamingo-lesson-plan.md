# Multimodal Lesson 06 Flamingo-Lite Implementation Plan

**Goal:** Add `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm` as a teaching lesson for interleaved image-text few-shot multimodal prompting.

**Architecture:** The lesson will synthesize two support demonstrations plus one query in a single prompt containing `<image>` placeholders. A tiny CNN will encode each image, the model will inject image embeddings into the text stream at aligned `<image>` positions, and a small decoder-style GRU will predict the query answer suffix.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock discovery and lesson contract with red tests

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_flamingo.py`

**Step 1: Write the failing test**

Add tests that require:

- `lesson_06_flamingo_compact_interleaved_vlm` to appear in `python scripts/run_lesson.py multimodal --list`
- dry-run resolution to `tracks.multimodal.lesson_06_flamingo_compact_interleaved_vlm.train`
- a focused lesson test module for data, model, and training smoke

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_flamingo.py`

Expected: FAIL because lesson 6 files do not exist yet.

**Step 3: Write minimal implementation**

Create only the package shell needed for the next failure.

**Step 4: Run test to verify the next failure**

Re-run the same command and use the next failure as the next target.

### Task 2: Build the synthetic interleaved prompt dataset

**Files:**
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/data.py`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/README.md`
- Test: `tests/test_tracks_multimodal_flamingo.py`

**Step 1: Write the failing test**

Require:

- `images` shaped `(B, 3, 3, H, W)`
- `prompt_ids`, `input_ids`, `labels`, `attention_mask` with consistent `(B, T)` shapes
- vocab entries for `<image>`, `example`, `query`, `dax`, `blicket`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_flamingo.py::test_multimodal_flamingo_batch_shapes`

Expected: FAIL because `data.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- lesson-local vocab
- support/query interleaved prompt construction
- synthetic task-token-to-attribute supervision
- collate and dataloader helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 3: Implement the Flamingo-lite model

**Files:**
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/model.py`
- Test: `tests/test_tracks_multimodal_flamingo.py`

**Step 1: Write the failing test**

Require:

- model outputs include `logits` and `image_embeddings`
- logits shaped `(B, T, vocab_size)`
- finite QA loss

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_flamingo.py::test_multimodal_flamingo_model_outputs`

Expected: FAIL because `model.py` is missing.

**Step 3: Write minimal implementation**

Implement:

- tiny multi-image vision encoder
- image-slot injection into the token stream
- decoder-style GRU over the interleaved prompt
- QA loss and exact-match helpers

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 4: Add the training entrypoint and smoke run

**Files:**
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/train.py`
- Modify: `tracks/multimodal/README.md`
- Test: `tests/test_tracks_multimodal_flamingo.py`

**Step 1: Write the failing test**

Add a subprocess smoke test that runs:

`python -m tracks.multimodal.lesson_06_flamingo_compact_interleaved_vlm.train --epochs 1 --num-samples 64 --batch-size 8 --image-size 16 --max-text-length 28 --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name pytest_flamingo_smoke`

Assert that it exits successfully and writes:

- `outputs/multimodal/lesson_06_flamingo_compact_interleaved_vlm/pytest_flamingo_smoke/config.json`
- `outputs/multimodal/lesson_06_flamingo_compact_interleaved_vlm/pytest_flamingo_smoke/metrics.jsonl`
- `outputs/multimodal/lesson_06_flamingo_compact_interleaved_vlm/pytest_flamingo_smoke/checkpoints/checkpoint.pt`

**Step 2: Run test to verify it fails**

Run:

`pytest -q tests/test_tracks_multimodal_flamingo.py::test_multimodal_flamingo_training_smoke`

Expected: FAIL because `train.py` does not exist yet.

**Step 3: Write minimal implementation**

Implement:

- CLI parsing
- train/eval loops
- sample logging with prompt and answers
- config, vocab, metrics, checkpoint writing
- track README update for lesson 6

**Step 4: Run test to verify it passes**

Run the same test and expect PASS.

### Task 5: Verify the feature

**Files:**
- Modify: `tests/test_scripts_run_lesson.py`
- Create: `tests/test_tracks_multimodal_flamingo.py`
- Modify: `tracks/multimodal/README.md`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/__init__.py`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/data.py`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/model.py`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/train.py`
- Create: `tracks/multimodal/lesson_06_flamingo_compact_interleaved_vlm/README.md`

**Step 1: Run lint**

Run:

`ruff check tracks/multimodal tests/test_tracks_multimodal_flamingo.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_clip.py tests/test_scripts_run_lesson.py docs/plans/2026-03-14-multimodal-flamingo-lesson-design.md docs/plans/2026-03-14-multimodal-flamingo-lesson-plan.md`

Expected: PASS.

**Step 2: Run targeted tests**

Run:

`pytest -q tests/test_scripts_run_lesson.py tests/test_tracks_multimodal_clip.py tests/test_tracks_multimodal_blip.py tests/test_tracks_multimodal_llava.py tests/test_tracks_multimodal_grounding.py tests/test_tracks_multimodal_mask_grounding.py tests/test_tracks_multimodal_flamingo.py`

Expected: PASS.

**Step 3: Run manual discovery smoke**

Run:

- `python scripts/run_lesson.py multimodal --list`
- `python scripts/run_lesson.py multimodal lesson_06_flamingo_compact_interleaved_vlm --dry-run`

Expected: lesson 6 appears in the listing and resolves to the train module.
