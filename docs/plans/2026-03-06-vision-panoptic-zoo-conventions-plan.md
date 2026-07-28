# Vision Panoptic Zoo Conventions Implementation Plan

**Goal:** Bring the existing `dlhub/vision/panoptic_segmentation/` 40-family package up to the same discoverable local-zoo standard as detection, instance segmentation, and FGVC.

**Architecture:** Keep the existing per-family panoptic implementations unchanged. Add a lazy AST-discovered `panoptic_segmentation_zoo` registry that enumerates `_VARIANTS`, builds models by `dlpan:*` arch id, and filters kwargs against each builder signature. Add a small CLI script and pytest smoke coverage for list/build/backward flows across all tiny variants plus representative non-tiny variants.

**Tech Stack:** Python, PyTorch, pytest, AST parsing, subprocess CLI smoke tests.

---

### Task 1: Add failing panoptic zoo tests

**Files:**
- Create: `tests/test_dlhub_vision_panoptic_segmentation_zoo.py`

**Step 1: Write the failing test**

Add a zoo test that:
- imports `dlhub.vision.panoptic_segmentation_zoo`
- asserts `len(list_local_arches()) >= 120`
- asserts representative ids exist:
  - `dlpan:panoptic_fpn_tiny`
  - `dlpan:mask2former_panoptic_tiny`
  - `dlpan:panoptic_deeplab_tiny`
  - `dlpan:transunet_panoptic_tiny`
  - `dlpan:rtdetr_panoptic_tiny`
- parametrizes over all `dlpan:*_tiny` ids and runs forward/backward smoke
- builds representative `small/base` variants
- subprocess-smokes `scripts/panoptic_segmentation_zoo.py --list` and `--smoke`

**Step 2: Run test to verify it fails**

Run:
- `pytest -q tests/test_dlhub_vision_panoptic_segmentation_zoo.py`

Expected:
- import error for missing zoo module or script

### Task 2: Add lazy panoptic zoo registry and CLI

**Files:**
- Create: `dlhub/vision/panoptic_segmentation_zoo.py`
- Create: `scripts/panoptic_segmentation_zoo.py`

**Step 1: Use Task 1 as RED**

Do not modify panoptic family files.

**Step 2: Write minimal implementation**

Add:
- `BuildConfig` with `in_channels`, `num_thing_classes`, `num_stuff_classes`, `width_mult`
- lazy AST discovery of `_VARIANTS` and `build_*_panoptic_segmenter`
- `list_local_arches()` returning `dlpan:*`
- `build_local_model(...)` with prefix validation and signature-aware kwargs filtering
- CLI with `--list`, `--search`, `--smoke`

**Step 3: Run test to verify it passes**

Run:
- `pytest -q tests/test_dlhub_vision_panoptic_segmentation_zoo.py`

### Task 3: Full verification

**Files:**
- Modify as needed from failures

**Step 1: Run targeted regression**

Run:
- `pytest -q tests/test_dlhub_vision_panoptic_segmentation_algorithms.py tests/test_dlhub_vision_panoptic_segmentation_zoo.py`

**Step 2: Run full regression**

Run:
- `pytest -q`

**Step 3: Keep work isolated**

Do not push from the main workspace. Keep changes in the existing worktree branch.
