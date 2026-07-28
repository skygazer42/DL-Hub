# Vision Detection 50 Algorithms Implementation Plan

**Goal:** Add 50 new real object detection families to `dlhub/vision/detection/`, grouped by timeline and detector class, with smokeable builders, timeline CLI support, and test coverage.

**Architecture:** Keep the existing file-per-family detection layout and AST-discovered builders. Add a timeline metadata layer for `2014-2026`, extend the detection CLI with `--timeline`, and implement new detector family modules in category batches using shared compact-first building blocks from `dlhub/vision/detection/_common.py`.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing `dlhub` detection helpers and lazy import conventions.

---

### Task 1: Establish the Failing Test Surface

**Files:**
- Modify: `tests/test_dlhub_vision_detection_zoo.py`
- Modify: `tests/test_dlhub_vision_detection_algorithms.py`
- Create: `tests/test_dlhub_vision_detection_timeline.py`

**Step 1: Write the failing tests**

- Extend the zoo test to assert representative new `arch_id`s from all five groups.
- Add a timeline test that imports `dlhub.vision.detection._timeline`, validates year/group metadata, and exercises `python scripts/detection_zoo.py --timeline`.
- Extend the algorithm smoke test with representative builders from the new families.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_timeline.py tests/test_dlhub_vision_detection_algorithms.py
```

Expected:

- missing metadata module and timeline CLI support
- missing new builder exports
- missing new families in the zoo

**Step 3: Write minimal implementation**

- Do not add family modules yet.
- First only add enough metadata plumbing and test scaffolding so the failure surface is precise.

**Step 4: Run the focused tests again**

Run the same pytest command and confirm the failures now point only at missing implementations.

**Step 5: Commit**

Commit message:

```bash
git commit -m "test(vision): add detection timeline and archive expectations"
```

### Task 2: Add Detection Timeline Metadata and CLI Support

**Files:**
- Create: `dlhub/vision/detection/_timeline.py`
- Modify: `scripts/detection_zoo.py`

**Step 1: Write the failing test**

- Use the tests from Task 1 as the red state.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_timeline.py
```

Expected:

- import failures or missing `--timeline` output

**Step 3: Write minimal implementation**

- Add `TimelineEntry` dataclass and `entries()` / `by_family()` helpers.
- Populate metadata for all existing and planned detection families.
- Add `--timeline` support to `scripts/detection_zoo.py` with grouped yearly output and example `arch_id`s.

**Step 4: Run test to verify GREEN**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_timeline.py
```

Expected:

- PASS

**Step 5: Commit**

Commit message:

```bash
git commit -m "feat(vision): add detection timeline metadata and CLI"
```

### Task 3: Implement Batch A Single-stage Families

**Files:**
- Create: `dlhub/vision/detection/overfeat.py`
- Create: `dlhub/vision/detection/ron.py`
- Create: `dlhub/vision/detection/refinedet.py`
- Create: `dlhub/vision/detection/m2det.py`
- Create: `dlhub/vision/detection/yolov4.py`
- Create: `dlhub/vision/detection/ppyolo.py`
- Create: `dlhub/vision/detection/scaled_yolov4.py`
- Create: `dlhub/vision/detection/ppyolov2.py`

**Step 1: Write the failing test**

- Add representative builder names and `arch_id`s for this batch to the detection smoke tests.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_algorithms.py -k "overfeat or ron or refinedet or m2det or yolov4 or ppyolo or scaled_yolov4 or ppyolov2"
```

Expected:

- missing builders

**Step 3: Write minimal implementation**

- Reuse single-stage conv/FPN helpers where possible.
- Include `_VARIANTS`, `build_*_detector`, and `__main__` smoke in every file.

**Step 4: Run test to verify GREEN**

Run the same pytest command and confirm it passes.

**Step 5: Commit**

Commit message:

```bash
git commit -m "feat(vision): add batch-a single-stage detection families"
```

### Task 4: Implement Batch B Single-stage Families

**Files:**
- Create: `dlhub/vision/detection/giraffedet.py`
- Create: `dlhub/vision/detection/yolov6.py`
- Create: `dlhub/vision/detection/yolov7.py`
- Create: `dlhub/vision/detection/damo_yolo.py`
- Create: `dlhub/vision/detection/gold_yolo.py`
- Create: `dlhub/vision/detection/yolov9.py`
- Create: `dlhub/vision/detection/yolov10.py`

**Step 1: Write the failing test**

- Extend focused smoke coverage for this batch.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_algorithms.py -k "giraffedet or yolov6 or yolov7 or damo_yolo or gold_yolo or yolov9 or yolov10"
```

**Step 3: Write minimal implementation**

- Keep these models compatible with the current AST discovery scheme.

**Step 4: Run test to verify GREEN**

Run the same command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat(vision): add batch-b modern single-stage detection families"
```

### Task 5: Implement Batch C Two-stage Families

**Files:**
- Create: `dlhub/vision/detection/rcnn.py`
- Create: `dlhub/vision/detection/sppnet.py`
- Create: `dlhub/vision/detection/fast_rcnn.py`
- Create: `dlhub/vision/detection/hypernet.py`
- Create: `dlhub/vision/detection/tridentnet.py`
- Create: `dlhub/vision/detection/libra_rcnn.py`
- Create: `dlhub/vision/detection/grid_rcnn.py`
- Create: `dlhub/vision/detection/guided_anchoring_rcnn.py`
- Create: `dlhub/vision/detection/detectors.py`
- Create: `dlhub/vision/detection/dynamic_rcnn.py`

**Step 1: Write the failing test**

- Add representative builder expectations for this batch.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_algorithms.py -k "rcnn or sppnet or fast_rcnn or hypernet or tridentnet or libra_rcnn or grid_rcnn or guided_anchoring_rcnn or detectors or dynamic_rcnn"
```

**Step 3: Write minimal implementation**

- Reuse proposal-style helpers rather than duplicating region pooling scaffolding.

**Step 4: Run test to verify GREEN**

Run the same command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat(vision): add batch-c two-stage detection families"
```

### Task 6: Implement Batch D Keypoint / Anchor-free Families

**Files:**
- Create: `dlhub/vision/detection/densebox.py`
- Create: `dlhub/vision/detection/unitbox.py`
- Create: `dlhub/vision/detection/point_linking_network.py`
- Create: `dlhub/vision/detection/borderdet.py`
- Create: `dlhub/vision/detection/autoassign.py`
- Create: `dlhub/vision/detection/centernet2.py`

**Step 1: Write the failing test**

- Extend representative smoke coverage for this batch.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_algorithms.py -k "densebox or unitbox or point_linking_network or borderdet or autoassign or centernet2"
```

**Step 3: Write minimal implementation**

- Keep outputs tensor-like and smoke-friendly.

**Step 4: Run test to verify GREEN**

Run the same command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat(vision): add batch-d keypoint and anchor-free detection families"
```

### Task 7: Implement Batch E Transformer / Query Families

**Files:**
- Create: `dlhub/vision/detection/anchor_detr.py`
- Create: `dlhub/vision/detection/smca_detr.py`
- Create: `dlhub/vision/detection/yolos.py`
- Create: `dlhub/vision/detection/adamixer.py`
- Create: `dlhub/vision/detection/efficient_detr.py`
- Create: `dlhub/vision/detection/deta.py`
- Create: `dlhub/vision/detection/h_detr.py`
- Create: `dlhub/vision/detection/co_detr.py`
- Create: `dlhub/vision/detection/group_detr.py`
- Create: `dlhub/vision/detection/ddq_detr.py`

**Step 1: Write the failing test**

- Extend representative smoke coverage for this batch.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_algorithms.py -k "anchor_detr or smca_detr or yolos or adamixer or efficient_detr or deta or h_detr or co_detr or group_detr or ddq_detr"
```

**Step 3: Write minimal implementation**

- Reuse query-decoder helpers and keep outputs aligned with current DETR-like families.

**Step 4: Run test to verify GREEN**

Run the same command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat(vision): add batch-e transformer and query detection families"
```

### Task 8: Implement Batch F Open-vocabulary / Multimodal Families

**Files:**
- Create: `dlhub/vision/detection/vild.py`
- Create: `dlhub/vision/detection/regionclip.py`
- Create: `dlhub/vision/detection/glip.py`
- Create: `dlhub/vision/detection/detclip.py`
- Create: `dlhub/vision/detection/owl_vit.py`
- Create: `dlhub/vision/detection/grounding_dino.py`
- Create: `dlhub/vision/detection/detclipv2.py`
- Create: `dlhub/vision/detection/yolo_world.py`
- Create: `dlhub/vision/detection/decola.py`

**Step 1: Write the failing test**

- Extend representative smoke coverage for this batch, including at least one text-conditioned path.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_algorithms.py -k "vild or regionclip or glip or detclip or owl_vit or grounding_dino or detclipv2 or yolo_world or decola"
```

**Step 3: Write minimal implementation**

- Keep text conditioning optional so generic smoke paths still work without downloads.

**Step 4: Run test to verify GREEN**

Run the same command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat(vision): add batch-f open-vocabulary detection families"
```

### Task 9: Update Detection Exports and Archive Coverage

**Files:**
- Modify: `dlhub/vision/detection/__init__.py`
- Modify: `dlhub/vision/detection_zoo.py`
- Modify: `scripts/detection_zoo.py`
- Modify: `tests/test_zoo_conventions_smoke.py`

**Step 1: Write the failing test**

- Extend archive expectations to the new families and convention coverage.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_zoo_conventions_smoke.py tests/test_dlhub_vision_detection_zoo.py
```

**Step 3: Write minimal implementation**

- Ensure lazy imports and AST discovery still work with all added files.

**Step 4: Run test to verify GREEN**

Run the same command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat(vision): wire expanded detection archive into zoo and conventions"
```

### Task 10: Final Verification

**Files:**
- Verify only; no new files required.

**Step 1: Run focused detection verification**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_timeline.py tests/test_dlhub_vision_detection_algorithms.py tests/test_zoo_conventions_smoke.py
```

**Step 2: Run broader regression coverage**

Run:

```bash
pytest -q tests/test_dlhub_vision_detection_zoo.py tests/test_dlhub_vision_detection_timeline.py tests/test_dlhub_vision_detection_algorithms.py tests/test_zoo_conventions_smoke.py tests/test_repo_layout.py
```

**Step 3: Inspect actual output**

- Record pass/fail status honestly.
- If failures remain, stop claiming completion and report the exact failing area.

**Step 4: Commit**

```bash
git commit -m "test(vision): verify expanded detection archive"
```
