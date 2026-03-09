# PointCloud Tracking3D Zoo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first 3D object tracking zoo for point-cloud/LiDAR-style tracking families with discovery, timeline metadata, CLI utilities, and smoke coverage.

**Architecture:** Create a new `dlhub/pointcloud/tracking3d/` package parallel to `detection3d`, with one tracking family per file. Add a lightweight `tracking3d_zoo.py` registry, `scripts/tracking3d_zoo.py` CLI, best-effort timeline metadata, and toy-first trackers that operate on short point-cloud sequences and return stable track tensors.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing `dlhub.pointcloud` zoo conventions.

---

### Task 1: Add the failing 3D tracking test surface

**Files:**
- Create: `tests/test_dlhub_pointcloud_tracking3d_timeline.py`
- Create: `tests/test_dlhub_pointcloud_tracking3d_zoo.py`
- Create: `tests/test_dlhub_pointcloud_tracking3d_algorithms.py`
- Modify: `tests/test_zoo_conventions_smoke.py`

**Step 1: Write the failing test**

- Assert a new zoo lists representative ids for `ab3dmot`, `centerpoint_track`, `simpletrack`, `bitrack`, `motsf3d`, `imm_kalman`.
- Assert the timeline covers those families with year/group metadata.
- Assert each representative builder can run a short sequence smoke and returns finite tensors.
- Add the new tracking family directory to the convention audit.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_dlhub_pointcloud_tracking3d_timeline.py tests/test_dlhub_pointcloud_tracking3d_zoo.py tests/test_dlhub_pointcloud_tracking3d_algorithms.py
```

Expected:
- import failures for missing tracking package/zoo/timeline

**Step 3: Write minimal implementation**

- Do not add lesson code.
- Only add enough scaffolding so failures point at missing trackers/builders.

**Step 4: Run the focused tests again**

Run the same pytest command to confirm the failure surface is precise.

**Step 5: Commit**

```bash
git commit -m "test: add tracking3d zoo expectations"
```

### Task 2: Add tracking3d package core, timeline, and CLI

**Files:**
- Create: `dlhub/pointcloud/tracking3d/__init__.py`
- Create: `dlhub/pointcloud/tracking3d/_common.py`
- Create: `dlhub/pointcloud/tracking3d/_timeline.py`
- Create: `dlhub/pointcloud/tracking3d_zoo.py`
- Create: `scripts/tracking3d_zoo.py`

**Step 1: Write the failing test**

- Reuse the red tests from Task 1.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_pointcloud_tracking3d_timeline.py tests/test_dlhub_pointcloud_tracking3d_zoo.py
```

**Step 3: Write minimal implementation**

- Add a sequence-oriented `BuildConfig`.
- Add shared helpers for checking `(B,T,N,C)` point-cloud sequences and producing toy track tensors.
- Add best-effort timeline metadata and CLI support for `--list`, `--timeline`, `--smoke`.

**Step 4: Run test to verify GREEN**

Run the same pytest command until it passes.

**Step 5: Commit**

```bash
git commit -m "feat: add tracking3d zoo core"
```

### Task 3: Implement the first tracking3d families

**Files:**
- Create: `dlhub/pointcloud/tracking3d/ab3dmot.py`
- Create: `dlhub/pointcloud/tracking3d/centerpoint_track.py`
- Create: `dlhub/pointcloud/tracking3d/simpletrack.py`
- Create: `dlhub/pointcloud/tracking3d/bitrack.py`
- Create: `dlhub/pointcloud/tracking3d/motsf3d.py`
- Create: `dlhub/pointcloud/tracking3d/imm_kalman.py`

**Step 1: Write the failing test**

- Use the representative algorithm smoke tests from Task 1.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_pointcloud_tracking3d_algorithms.py
```

**Step 3: Write minimal implementation**

- Keep trackers toy-first and sequence-first.
- Each file must expose `_VARIANTS`, `build_*_tracker3d(...)`, and a `__main__` smoke path.
- Reuse shared sequence tracking helpers instead of duplicating association/motion scaffolding.

**Step 4: Run test to verify GREEN**

Run:

```bash
pytest -q tests/test_dlhub_pointcloud_tracking3d_timeline.py tests/test_dlhub_pointcloud_tracking3d_zoo.py tests/test_dlhub_pointcloud_tracking3d_algorithms.py tests/test_zoo_conventions_smoke.py -k tracking3d
```

**Step 5: Commit**

```bash
git commit -m "feat: add first pointcloud tracking3d strategy batch"
```
