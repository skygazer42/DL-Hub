# Vision Detection (40 Algorithms) Implementation Plan

**Goal:** Expand `dlhub/vision/detection/` into a **40-algorithm compact-first object detection suite** (pure PyTorch), with smoke tests and a push to `main`.

**Architecture:** Keep **one algorithm family per file** in `dlhub/vision/detection/` (variants live inside the same file via `_VARIANTS` + `build_*` factory). Each algorithm file includes a `__main__` random forward/backward smoke. A single pytest smoke iterates all builders and validates forward/backward on CPU.

**Tech Stack:** Python 3.10+, PyTorch (no external detection packages), existing `dlhub` Conv blocks, pytest.

---

## Scope / Definition

- “40 algorithms” here means **40 detector algorithm families** available under `dlhub.vision.detection` (including the already-present FCOS / CenterNet / RetinaNet / YOLOv1).
- Implementations are **compact-first** and focus on architecture/forward pass + stability (forward/backward), not full COCO-grade training pipelines.

## Algorithm List (Target = 40 families)

Already present (4):
- `fcos.py` (FCOS)
- `centernet.py` (CenterNet)
- `retinanet.py` (RetinaNet)
- `yolo.py` (YOLOv1)

Planned additions (36):
- One-stage / anchors & FPN-style:
  - `ssd.py` (SSD)
  - `dssd.py` (DSSD)
  - `efficientdet.py` (EfficientDet / BiFPN-style)
  - `squeezedet.py` (SqueezeDet-style)
  - `atss.py` (ATSS-style)
  - `paa.py` (PAA-style)
  - `freeanchor.py` (FreeAnchor-style)
  - `fsaf.py` (FSAF-style)
- YOLO family:
  - `yolov2.py` (YOLOv2-style)
  - `yolov3.py` (YOLOv3-style, multi-scale)
  - `yolov5.py` (YOLOv5-style, CSP + PAN)
  - `yolox.py` (YOLOX-style, decoupled head)
  - `yolof.py` (YOLOF-style, single-level + dilated encoder)
  - `ppyoloe.py` (PP-YOLOE-style)
  - `rtmdet.py` (RTMDet-style)
- Quality / distribution regression family:
  - `nanodet.py` (NanoDet / GFL-style)
  - `gfl.py` (GFL-style)
  - `tood.py` (TOOD-style)
  - `varifocalnet.py` (VarifocalNet-style)
  - `vfnet.py` (VFNet-style)
  - `reppoints.py` (RepPoints-style)
  - `foveabox.py` (FoveaBox-style)
- Keypoint-based:
  - `cornernet.py` (CornerNet-style)
  - `extremenet.py` (ExtremeNet-style)
  - `ttfnet.py` (TTFNet-style)
- Transformer / query-based:
  - `detr.py` (DETR)
  - `deformable_detr.py` (Deformable DETR *style*)
  - `conditional_detr.py` (Conditional DETR *style*)
  - `dab_detr.py` (DAB-DETR *style*)
  - `dn_detr.py` (DN-DETR *style*)
  - `rtdetr.py` (RT-DETR *style*)
  - `sparse_rcnn.py` (Sparse R-CNN *style*)
- Two-stage:
  - `faster_rcnn.py` (Faster R-CNN *compact*)
  - `mask_rcnn.py` (Mask R-CNN *compact*)
  - `cascade_rcnn.py` (Cascade R-CNN *compact*)
  - `rfcn.py` (R-FCN *compact*)

## Task 1: Add/Update Test Harness for Detection Algorithms

**Files:**
- Create: `tests/test_dlhub_vision_detection_algorithms.py`

**Step 1: Write failing test**
- The test imports `dlhub.vision.detection` and enumerates all `build_*_detector` factories for the 40 algorithms.
- For each builder:
  - instantiate with small config (e.g. `in_channels=3`, `num_classes=2`, `variant=...`, `width_mult=0.5`)
  - run forward on `torch.randn(2, 3, 64, 64)`
  - compute a scalar loss by recursively summing tensor means from dict/list outputs
  - run `backward()`

**Step 2: Run to verify RED**
- Run: `pytest -q tests/test_dlhub_vision_detection_algorithms.py`
- Expected: FAIL because new modules/builders do not exist yet.

## Task 2: Implement Detector Modules (Batch-by-batch)

For each algorithm file:
- Create: `dlhub/vision/detection/<algo>.py`
- Add `_VARIANTS` with at least `*_tiny/small/base`
- Add `build_<algo>_detector(...)`
- Add `if __name__ == "__main__":` smoke (forward + backward + print)

Execute in batches (suggested commits per batch):

### Batch A (foundation one-stage)
- `ssd.py`, `dssd.py`, `efficientdet.py`, `squeezedet.py`

### Batch B (YOLO family)
- `yolov2.py`, `yolov3.py`, `yolov5.py`, `yolox.py`, `yolof.py`, `ppyoloe.py`, `rtmdet.py`

### Batch C (quality/distribution/assignment styles)
- `nanodet.py`, `gfl.py`, `tood.py`, `varifocalnet.py`, `vfnet.py`
- `atss.py`, `paa.py`, `freeanchor.py`, `fsaf.py`
- `reppoints.py`, `foveabox.py`

### Batch D (keypoint-based)
- `cornernet.py`, `extremenet.py`, `ttfnet.py`

### Batch E (transformer/query-based)
- `detr.py`, `deformable_detr.py`, `conditional_detr.py`, `dab_detr.py`, `dn_detr.py`, `rtdetr.py`, `sparse_rcnn.py`

### Batch F (two-stage compact)
- `faster_rcnn.py`, `mask_rcnn.py`, `cascade_rcnn.py`, `rfcn.py`

After each batch:
- Run: `pytest -q tests/test_dlhub_vision_detection_algorithms.py`
- Commit with message: `feat(vision): add <batch> detection models`

## Task 3: Update Detection Exports

**Files:**
- Modify: `dlhub/vision/detection/__init__.py`

**Steps:**
- Export all new `build_*_detector` functions.
- Prefer lazy imports for builders to keep `import dlhub.vision.detection` lightweight.

## Task 4: Docs

**Files:**
- Modify: `docs/STRUCTURE.md` (only if needed)

Optional:
- Update `tracks/vision/README.md` to mention the expanded detection library (no need to add 36 lessons).

## Task 5: Verification + Push

**Step 1: Full test suite**
- Run: `pytest -q`
- Expected: PASS

**Step 2: Commit and push**
- Commit any remaining changes
- Run: `git push origin main`
