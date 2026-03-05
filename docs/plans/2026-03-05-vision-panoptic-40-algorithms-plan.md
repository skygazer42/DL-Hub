# Vision Panoptic Segmentation Zoo (40 Families) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pure-torch, toy-first panoptic segmentation zoo with ~40 algorithm families, each implemented as a full `nn.Module` in a single file with variants and a `__main__` smoke test, plus pytest forward/backward smoke coverage.

**Architecture:** Create a dedicated package `dlhub/vision/panoptic_segmentation/` with a shared `_common.py` (tiny backbones, FPN, prototype masks, and panoptic fusion). Each algorithm family lives in its own file and exposes `build_<name>_panoptic_segmenter(...)`. Add a single pytest that instantiates each builder on a small input and runs forward + backward.

**Tech Stack:** Python, PyTorch (`torch`, `torch.nn`, `torch.nn.functional`), pytest.

---

### Task 1: Stabilize the new panoptic package

**Files:**
- Create/Modify: `dlhub/vision/panoptic_segmentation/_common.py`
- Modify: `dlhub/vision/panoptic_segmentation/__init__.py`

**Step 1: Write/adjust a failing import smoke**
- Test file: `tests/test_dlhub_vision_panoptic_segmentation_algorithms.py`
- Expected RED: importing `dlhub.vision.panoptic_segmentation` fails while modules are missing.

**Step 2: Make imports robust**
- Keep `__init__.py` only importing implemented families until all files exist.
- Ensure `_common.py` exposes the minimal shared blocks used across families.

**Step 3: Verify GREEN**
- Run: `pytest tests/test_dlhub_vision_panoptic_segmentation_algorithms.py -q`
- Expected: still fails (missing algorithms), but fails for the “right reason” (missing modules/builders), not syntax errors.

---

### Task 2: Implement panoptic families (one algorithm family per file)

**Files:**
- Create: `dlhub/vision/panoptic_segmentation/<algorithm>.py` (many)
- Modify: `dlhub/vision/panoptic_segmentation/__init__.py`

**Algorithm file contract (repeatable template):**
- `class <AlgoName>(nn.Module):` full network definition
- `_VARIANTS: dict[str, dict]` (e.g. `*_tiny/*_small/*_base`)
- `def build_<algo>_panoptic_segmenter(...):` factory
- `if __name__ == "__main__":` random forward + backward smoke

**Step 1: RED**
- Add each algorithm name to the pytest parametrization list and run the test to confirm it fails because the file/builder doesn’t exist yet.

**Step 2: GREEN**
- Implement the file with a toy-first architecture that produces at least:
  - `semantic_logits` (B, num_thing+num_stuff, H, W)
  - `mask_logits` (B, N, H, W)
  - an instance confidence signal (e.g. `query_cls_logits` or `instance_scores`)
  - optional convenience `panoptic_map` via `_common.fuse_panoptic`

**Step 3: Verify**
- Run targeted smoke:
  - `python -m dlhub.vision.panoptic_segmentation.<algorithm_module>`
- Then run the pytest again.

---

### Task 3: Add pytest smoke coverage for all families

**Files:**
- Create: `tests/test_dlhub_vision_panoptic_segmentation_algorithms.py`

**Step 1: RED**
- Parametrize over all build functions; expect failures until algorithms exist.

**Step 2: GREEN**
- For each builder:
  - instantiate with `in_channels=3`, `num_thing_classes=3`, `num_stuff_classes=2`, `variant=..._tiny`, `width_mult=0.5`
  - run forward on `x = torch.randn(2, 3, 64, 64)`
  - compute scalar loss by recursively summing tensor means
  - `loss.backward()`

**Step 3: Verify**
- Run: `pytest tests/test_dlhub_vision_panoptic_segmentation_algorithms.py -q`

---

### Task 4: Full verification + ship

**Files:**
- Modify as needed from failures

**Step 1: Run full tests**
- Run: `pytest -q`
- Expected: exit code 0

**Step 2: Commit**
- Run:
  - `git add dlhub/vision/panoptic_segmentation tests docs/plans/2026-03-05-vision-panoptic-40-algorithms-plan.md`
  - `git commit -m "feat(vision): add panoptic segmentation zoo (40 families)"`

**Step 3: Push**
- Run: `git push origin main`

