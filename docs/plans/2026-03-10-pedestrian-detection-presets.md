# Pedestrian Detection Presets Implementation Plan

**Goal:** Add 8 “pedestrian detection” preset arch ids to the local detection zoo (`dldet:pedestrian_*`) that forward/backward smoke successfully and can be discovered via `--search pedestrian`.

**Architecture:** Each preset is a tiny alias wrapper around an existing detector family (FCOS/RetinaNet/Faster R-CNN/SSD/YOLO/RT-DETR). The presets live as 8 independent `dlhub/vision/detection/pedestrian_*.py` modules exposing `_VARIANTS` + `build_*_detector`, so `dlhub.vision.detection_zoo` discovers them automatically by source scanning.

**Tech Stack:** Python 3.10+ style, PyTorch (`torch`), pytest, existing `dlhub.vision.detection` zoo conventions.

---

## Preflight Notes (Windows / repo-local)

- Work from repo root: `F:\\DL-Hub`
- Prefer lightweight verifications:
  - `python scripts/detection_zoo.py --list --search pedestrian`
  - `python scripts/detection_zoo.py --smoke dldet:pedestrian_fcos`
  - `pytest -q tests/test_dlhub_vision_pedestrian_detection_presets.py`

## Branch Strategy

You will end with **one integration branch** containing:

- 8 new preset modules
- 1 new pytest smoke file

And you will create **8 feature branches** (one module per branch), then merge them into the integration branch:

- Integration branch: `feat/pedestrian-detection-presets`
- Feature branches:
  - `feat/pedestrian-fcos`
  - `feat/pedestrian-retinanet`
  - `feat/pedestrian-faster-rcnn`
  - `feat/pedestrian-ssd`
  - `feat/pedestrian-yolov5`
  - `feat/pedestrian-yolov8`
  - `feat/pedestrian-yolox`
  - `feat/pedestrian-rtdetr`

---

### Task 1: (Optional) Prepare a `.worktrees/` directory for parallel branches

**Files:**
- Modify: `.gitignore`

**Step 1: Check if `.worktrees/` exists**

Run: `ls .worktrees`
Expected: either “not found” (ok) or listing (ok).

**Step 2: Ensure `.worktrees/` is ignored**

Run: `git check-ignore -q .worktrees; echo $?`
Expected: `0` if ignored.

If not ignored:

**Step 3: Add ignore rule**

Add a line to `.gitignore`:

```gitignore
/.worktrees/
```

**Step 4: Verify ignore works**

Run: `git check-ignore -q .worktrees; echo $?`
Expected: `0`

**Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore .worktrees directory"
```

---

### Task 2: Create integration branch and add the failing preset smoke test (TDD)

**Files:**
- Create: `tests/test_dlhub_vision_pedestrian_detection_presets.py`

**Step 1: Create integration branch**

Run: `git switch -c feat/pedestrian-detection-presets`
Expected: switched to new branch.

**Step 2: Write the failing test**

Create `tests/test_dlhub_vision_pedestrian_detection_presets.py`:

```python
import pytest

torch = pytest.importorskip("torch")


def _sum_tensor_means(x):
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((_sum_tensor_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((_sum_tensor_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type: {type(x)!r}")


@pytest.mark.parametrize(
    "arch_id",
    [
        "dldet:pedestrian_fcos",
        "dldet:pedestrian_retinanet",
        "dldet:pedestrian_faster_rcnn",
        "dldet:pedestrian_ssd",
        "dldet:pedestrian_yolov5",
        "dldet:pedestrian_yolov8",
        "dldet:pedestrian_yolox",
        "dldet:pedestrian_rtdetr",
    ],
)
def test_pedestrian_presets_forward_backward_smoke(arch_id: str) -> None:
    from dlhub.vision.detection_zoo import build_local_model

    model = build_local_model(arch_id, in_channels=3, num_classes=1, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    loss = _sum_tensor_means(out)
    assert torch.isfinite(loss)
    loss.backward()
```

**Step 3: Run the test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_pedestrian_detection_presets.py`
Expected: FAIL with `UnknownLocalArch` for `dldet:pedestrian_*`.

**Step 4: Commit the failing test**

```bash
git add tests/test_dlhub_vision_pedestrian_detection_presets.py
git commit -m "test: add pedestrian detection preset smokes (failing)"
```

---

## Preset Implementation Template (use in Tasks 3–10)

Each preset module is a tiny alias wrapper:

- Must define `_VARIANTS` with exactly 1 key (the arch id suffix).
- Must define a single `build_*_detector` function.
- Must include a `__main__` smoke that calls `smoke_aliased_detector(...)`.

Template:

```python
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.<base_module> import build_<base_stem>_detector as _build_base

_VARIANTS: dict[str, str] = {
    "<variant_key>": "<base_variant>",
}


def build_<variant_key>_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "<variant_key>",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="Pedestrian presets",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="<variant_key>",
        builder=build_<variant_key>_detector,
        variant="<variant_key>",
    )
```

---

### Task 3: Add `dldet:pedestrian_fcos` preset (feature branch)

**Files:**
- Create: `dlhub/vision/detection/pedestrian_fcos.py`

**Step 1: Create feature branch**

Run: `git switch -c feat/pedestrian-fcos`

**Step 2: Implement preset module**

Create `dlhub/vision/detection/pedestrian_fcos.py`:

```python
from torch import nn

from dlhub.vision.detection._aliases import build_aliased_detector, smoke_aliased_detector
from dlhub.vision.detection.fcos import build_fcos_detector as _build_base

_VARIANTS: dict[str, str] = {"pedestrian_fcos": "fcos_tiny"}


def build_pedestrian_fcos_detector(
    *,
    in_channels: int,
    num_classes: int,
    variant: str = "pedestrian_fcos",
    width_mult: float = 1.0,
) -> nn.Module:
    return build_aliased_detector(
        family="Pedestrian presets",
        variants=_VARIANTS,
        base_builder=_build_base,
        in_channels=in_channels,
        num_classes=num_classes,
        variant=variant,
        width_mult=width_mult,
    )


if __name__ == "__main__":
    smoke_aliased_detector(
        label="pedestrian_fcos",
        builder=build_pedestrian_fcos_detector,
        variant="pedestrian_fcos",
    )
```

**Step 3: Smoke via CLI**

Run: `python scripts/detection_zoo.py --smoke dldet:pedestrian_fcos --num-classes 1`
Expected: prints output summary, exits 0.

**Step 4: Run the new pytest**

Run: `pytest -q tests/test_dlhub_vision_pedestrian_detection_presets.py -k pedestrian_fcos`
Expected: PASS for that parametrized case (others may still fail).

**Step 5: Commit**

```bash
git add dlhub/vision/detection/pedestrian_fcos.py
git commit -m "feat: add pedestrian_fcos detection preset"
```

**Step 6: Merge into integration branch**

```bash
git switch feat/pedestrian-detection-presets
git merge --no-ff feat/pedestrian-fcos
```

---

### Task 4: Add `dldet:pedestrian_retinanet` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_retinanet.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-retinanet`
- Variant key: `pedestrian_retinanet`
- Base builder: `from dlhub.vision.detection.retinanet import build_retinanet_detector as _build_base`
- Base variant: `retinanet_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_retinanet --num-classes 1`
- Commit message: `feat: add pedestrian_retinanet detection preset`

---

### Task 5: Add `dldet:pedestrian_faster_rcnn` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_faster_rcnn.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-faster-rcnn`
- Variant key: `pedestrian_faster_rcnn`
- Base builder: `from dlhub.vision.detection.faster_rcnn import build_faster_rcnn_detector as _build_base`
- Base variant: `faster_rcnn_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_faster_rcnn --num-classes 1`
- Commit message: `feat: add pedestrian_faster_rcnn detection preset`

---

### Task 6: Add `dldet:pedestrian_ssd` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_ssd.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-ssd`
- Variant key: `pedestrian_ssd`
- Base builder: `from dlhub.vision.detection.ssd import build_ssd_detector as _build_base`
- Base variant: `ssd_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_ssd --num-classes 1`
- Commit message: `feat: add pedestrian_ssd detection preset`

---

### Task 7: Add `dldet:pedestrian_yolov5` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_yolov5.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-yolov5`
- Variant key: `pedestrian_yolov5`
- Base builder: `from dlhub.vision.detection.yolov5 import build_yolov5_detector as _build_base`
- Base variant: `yolov5_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_yolov5 --num-classes 1`
- Commit message: `feat: add pedestrian_yolov5 detection preset`

---

### Task 8: Add `dldet:pedestrian_yolov8` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_yolov8.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-yolov8`
- Variant key: `pedestrian_yolov8`
- Base builder: `from dlhub.vision.detection.yolov8 import build_yolov8_detector as _build_base`
- Base variant: `yolov8_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_yolov8 --num-classes 1`
- Commit message: `feat: add pedestrian_yolov8 detection preset`

---

### Task 9: Add `dldet:pedestrian_yolox` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_yolox.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-yolox`
- Variant key: `pedestrian_yolox`
- Base builder: `from dlhub.vision.detection.yolox import build_yolox_detector as _build_base`
- Base variant: `yolox_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_yolox --num-classes 1`
- Commit message: `feat: add pedestrian_yolox detection preset`

---

### Task 10: Add `dldet:pedestrian_rtdetr` preset

**Files:**
- Create: `dlhub/vision/detection/pedestrian_rtdetr.py`

**Steps:** Repeat Task 3 with:

- Branch: `feat/pedestrian-rtdetr`
- Variant key: `pedestrian_rtdetr`
- Base builder: `from dlhub.vision.detection.rtdetr import build_rtdetr_detector as _build_base`
- Base variant: `rtdetr_tiny`
- Smoke: `python scripts/detection_zoo.py --smoke dldet:pedestrian_rtdetr --num-classes 1`
- Commit message: `feat: add pedestrian_rtdetr detection preset`

---

### Task 11: Final verification on integration branch

**Files:**
- (Already created/merged in prior tasks)

**Step 1: Verify discoverability**

Run: `python scripts/detection_zoo.py --list --search pedestrian`
Expected: should print the 8 `dldet:pedestrian_*` entries.

**Step 2: Run targeted tests**

Run: `pytest -q tests/test_dlhub_vision_pedestrian_detection_presets.py`
Expected: PASS (or SKIP if torch missing).

**Step 3: Run full test suite (optional)**

Run: `pytest -q`
Expected: PASS (repo-specific; investigate only if failures relate to your changes).

**Step 4: Merge result**

At this point, all work lives on `feat/pedestrian-detection-presets` and is ready to merge further (e.g. to `main`) if desired.
