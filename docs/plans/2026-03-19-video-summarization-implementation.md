# Video Summarization Implementation Plan

**Goal:** Add a local compact-first video summarization zoo under `dlhub.vision` with six extractive summarization families, a CLI, and smoke tests.

**Architecture:** Reuse the repository's lazy zoo discovery pattern. Each family lives in its own file under `dlhub/vision/video_summarization/`, exposes `_VARIANTS` and `build_*_video_summarizer(...)`, and returns a dict with `scores` and `summary_mask` for `(B, T, C, H, W)` video input.

**Tech Stack:** Python, PyTorch, AST-based lazy registry discovery, pytest, argparse.

---

### Task 1: Add the failing zoo test

**Files:**
- Create: `F:/DL-Hub/tests/test_dlhub_vision_video_summarization_zoo.py`

**Step 1: Write the failing test**

Add tests that:
- expect at least 18 arches
- expect `vsum:dsn_tiny`, `vsum:sum_gan_small`, `vsum:cycle_sum_base`, `vsum:vasnet_tiny`, `vsum:dsnet_small`, `vsum:ca_sum_tiny`
- build representative families and check for `scores` and `summary_mask`
- run CLI `--list` and `--smoke`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_dlhub_vision_video_summarization_zoo.py -q`

Expected: fail because module and script do not exist yet.

### Task 2: Add shared video summarization utilities

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/_common.py`
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/__init__.py`

**Step 1: Write minimal shared utilities**

Add:
- input validation
- tiny frame encoder
- temporal conv / GRU helpers
- score-to-mask helper

**Step 2: Run the targeted test**

Run: `python -m pytest tests/test_dlhub_vision_video_summarization_zoo.py -q`

Expected: still fail on missing family modules and zoo.

### Task 3: Implement the first three families

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/dsn.py`
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/sum_gan.py`
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/cycle_sum.py`

**Step 1: Implement minimal families**

Each family should:
- define `_VARIANTS`
- expose `build_*_video_summarizer(...)`
- return `scores` and `summary_mask`

**Step 2: Run targeted tests**

Run: `python -m pytest tests/test_dlhub_vision_video_summarization_zoo.py -q`

Expected: partial failure because the zoo and remaining families are not present.

### Task 4: Implement the attention and proposal families

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/vasnet.py`
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/dsnet.py`
- Create: `F:/DL-Hub/dlhub/vision/video_summarization/ca_sum.py`

**Step 1: Implement minimal families**

Keep each family small:
- `vasnet`: temporal self-attention scorer
- `dsnet`: segment proposal scorer pooled back to frame scores
- `ca_sum`: content-attention scorer

**Step 2: Run targeted tests**

Run: `python -m pytest tests/test_dlhub_vision_video_summarization_zoo.py -q`

Expected: failure only from missing zoo/CLI pieces if model files are correct.

### Task 5: Add zoo discovery and CLI

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_summarization_zoo.py`
- Create: `F:/DL-Hub/scripts/video_summarization_zoo.py`

**Step 1: Implement lazy registry**

Mirror the style of `dlhub/vision/style_transfer_zoo.py` and `dlhub/vision/mot_zoo.py`.

**Step 2: Implement CLI**

Support:
- `--list`
- `--search`
- `--limit`
- `--smoke`

**Step 3: Run targeted tests**

Run: `python -m pytest tests/test_dlhub_vision_video_summarization_zoo.py -q`

Expected: pass.

### Task 6: Verify and document

**Files:**
- Modify: `F:/DL-Hub/docs/plans/2026-03-19-vision-video-summarization-design.md`

**Step 1: Run focused verification**

Run:
- `python -m pytest tests/test_dlhub_vision_video_summarization_zoo.py -q`
- `python scripts/video_summarization_zoo.py --list --limit 6`
- `python scripts/video_summarization_zoo.py --smoke vsum:vasnet_tiny`

**Step 2: Keep the design doc aligned with the final shipped families**

Adjust names or details if the implementation differs from the draft.

