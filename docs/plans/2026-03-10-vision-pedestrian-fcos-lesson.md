# Vision Pedestrian Detection (Synthetic FCOS) Implementation Plan

**Goal:** Add a new Vision lesson that trains `dldet:pedestrian_fcos` on an offline synthetic pedestrian dataset and logs stable metrics (loss / center-acc / IoU).

**Architecture:** Reuse the local detection zoo preset `dldet:pedestrian_fcos` (FCOS-style) and implement a stride-4 synthetic dataset that produces FCOS-like targets (cls + ltrb + centerness). Keep the lesson compact-first and CPU-friendly.

**Tech Stack:** Python 3.10+, PyTorch (`torch`), pytest, existing `dlhub.paths/build_run_paths` and local detection zoo (`dlhub.vision.detection_zoo`).

---

### Task 1: Add lesson scaffold (empty modules)

**Files:**
- Create: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/__init__.py`
- Create: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/README.md`
- Create: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/data.py`
- Create: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/model.py`
- Create: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/train.py`

**Step 1: Create the directory + placeholder files**

Create minimal files (even if functions are `pass`) so imports resolve.

**Step 2: Sanity-check module discovery**

Run: `python -c "import tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos"`
Expected: exit code `0`

**Step 3: Commit**

```bash
git add tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos
git commit -m "feat(vision): scaffold synthetic pedestrian detection lesson"
```

---

### Task 2: Write failing pytest smoke for the new lesson

**Files:**
- Create: `tests/test_tracks_vision_pedestrian_detection.py`

**Step 1: Write a failing test**

```python
import pytest

torch = pytest.importorskip("torch")


def test_vision_synth_pedestrian_fcos_shapes_and_loss_smoke() -> None:
    from tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.data import DataConfig, get_dataloaders
    from tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.model import ModelConfig, build_model

    train_loader, _ = get_dataloaders(
        DataConfig(
            num_samples=32,
            batch_size=4,
            image_size=64,
            stride=4,
            val_fraction=0.2,
            seed=0,
            num_workers=0,
            noise_std=0.1,
        )
    )
    x, targets = next(iter(train_loader))
    assert tuple(x.shape) == (4, 3, 64, 64)
    assert set(targets) >= {"cls_target", "reg_target", "centerness_target", "pos_mask", "box"}

    model = build_model(ModelConfig(arch="dldet:pedestrian_fcos", in_channels=3, num_classes=1, width_mult=0.5))
    out = model(x)
    assert set(out) >= {"cls_logits", "reg"}

    cls_logits = out["cls_logits"]
    reg = out["reg"]
    cls_target = targets["cls_target"]
    reg_target = targets["reg_target"]
    pos_mask = targets["pos_mask"]

    cls_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]))(cls_logits, cls_target)
    pred_pos = (reg * pos_mask).sum(dim=(2, 3))
    target_pos = (reg_target * pos_mask).sum(dim=(2, 3))
    reg_loss = torch.nn.SmoothL1Loss()(pred_pos, target_pos)

    loss = cls_loss + reg_loss
    assert torch.isfinite(loss)
```

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_tracks_vision_pedestrian_detection.py`
Expected: FAIL with `ModuleNotFoundError` or missing attributes.

**Step 3: Commit**

```bash
git add tests/test_tracks_vision_pedestrian_detection.py
git commit -m "test(vision): add synthetic pedestrian detection smoke (failing)"
```

---

### Task 3: Implement synthetic pedestrian dataset (`data.py`)

**Files:**
- Modify: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/data.py`

**Step 1: Implement `DataConfig`, `SyntheticPedestrianDetection`, `get_dataloaders`**

- Produce RGB images `(3,H,W)` with background noise + one tall rectangle.
- Produce targets:
  - `cls_target (1,Gh,Gw)` one-hot at the pedestrian center cell
  - `reg_target (4,Gh,Gw)` l/t/r/b at the positive cell
  - `centerness_target (1,Gh,Gw)` computed from ltrb at the positive cell
  - `pos_mask (1,Gh,Gw)` one-hot at the positive cell
  - `box (4,)` xyxy

**Step 2: Run the test**

Run: `pytest -q tests/test_tracks_vision_pedestrian_detection.py`
Expected: still FAIL (model not implemented yet), but dataset imports + shapes should work.

**Step 3: Commit**

```bash
git add tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/data.py
git commit -m "feat(vision): add synthetic pedestrian dataset for FCOS lesson"
```

---

### Task 4: Implement model wrapper (`model.py`)

**Files:**
- Modify: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/model.py`

**Step 1: Implement `ModelConfig`, `build_model`, `list_supported_arches`**

- Default `arch="dldet:pedestrian_fcos"`.
- Build via `dlhub.vision.detection_zoo.build_local_model`.

**Step 2: Run the test**

Run: `pytest -q tests/test_tracks_vision_pedestrian_detection.py`
Expected: FAIL only if train code is required; otherwise should PASS once model returns expected keys.

**Step 3: Commit**

```bash
git add tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/model.py
git commit -m "feat(vision): add model wrapper using detection zoo pedestrian preset"
```

---

### Task 5: Implement training script (`train.py`) + README

**Files:**
- Modify: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/train.py`
- Modify: `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/README.md`
- Modify: `tracks/vision/README.md`

**Step 1: Implement `train.py`**

- Follow the repo lesson template:
  - parse args into `TrainConfig` + `DataConfig` + `ModelConfig`
  - resolve device (`dlhub.device.resolve_device`)
  - build outputs via `dlhub.paths.build_run_paths`
  - log per-epoch metrics to jsonl
  - save checkpoint
- Loss:
  - `cls_loss` + `reg_loss` (+ optional `centerness_loss` if `out` contains `centerness`)
- Metrics:
  - `center_acc` and `mean_iou` (decode best cell bbox)

**Step 2: Update lesson docs**

- `tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos/README.md` with a CPU smoke command.
- Add the lesson to `tracks/vision/README.md`.

**Step 3: Run tests**

Run: `pytest -q tests/test_tracks_vision_pedestrian_detection.py`
Expected: PASS

Optional smoke:
- `python -m tracks.vision.lesson_13_synthetic_pedestrian_detection_fcos.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke`

**Step 4: Commit**

```bash
git add tracks/vision/lesson_13_synthetic_pedestrian_detection_fcos tracks/vision/README.md
git commit -m "feat(vision): add synthetic pedestrian detection FCOS lesson"
```

