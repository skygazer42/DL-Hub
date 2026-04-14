# Cross-Domain New Directions Batch 16 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 100 new toy-first algorithm families across 10 previously unimplemented directions spanning `vision`, `pointcloud`, `multimodal`, and `generative`, without adding lessons, dependencies, or pytest files.

**Architecture:** Create one package per new direction, one family per file, and one direction-local zoo or registry surface per direction. Keep each direction branch local-only, then land shared README and domain export wiring on one integration branch after all direction-level smoke checks pass.

**Tech Stack:** Python 3, `torch`, repo-local toy model helpers, lazy package `__init__.py` files, explicit registries for `vision` and `multimodal`, AST-discovery zoos for `pointcloud` and `generative`, `.worktrees/` git worktrees, and minimal import/list smoke verification.

---

### Task 1: Prepare the worktree lane layout

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/`
- Create: `F:/DL-Hub/.worktrees/batch160-integration/`

**Step 1: Verify the worktree root is ignored**

Run:

```powershell
git check-ignore -q .worktrees
```

Expected: exit code `0`.

**Step 2: Create the 10 direction worktrees plus the integration worktree**

Run:

```powershell
git worktree add .worktrees/batch160-image-deweathering -b batch160-image-deweathering
git worktree add .worktrees/batch160-transparent-depth-estimation -b batch160-transparent-depth-estimation
git worktree add .worktrees/batch160-pointcloud-forecasting -b batch160-pointcloud-forecasting
git worktree add .worktrees/batch160-pointcloud-anomaly-detection -b batch160-pointcloud-anomaly-detection
git worktree add .worktrees/batch160-video-text-retrieval -b batch160-video-text-retrieval
git worktree add .worktrees/batch160-embodied-question-answering -b batch160-embodied-question-answering
git worktree add .worktrees/batch160-audio-text-understanding -b batch160-audio-text-understanding
git worktree add .worktrees/batch160-text-to-video -b batch160-text-to-video
git worktree add .worktrees/batch160-video-to-video -b batch160-video-to-video
git worktree add .worktrees/batch160-world-models -b batch160-world-models
git worktree add .worktrees/batch160-integration -b batch160-integration
```

Expected: each worktree is created from the same starting commit.

**Step 3: Capture the clean baseline**

Run:

```powershell
git -C .worktrees/batch160-image-deweathering status --short --branch
git -C .worktrees/batch160-integration status --short --branch
```

Expected: each branch is shown with no tracked modifications.

### Task 2: Add the `image_deweathering` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/snow_removal.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/raindrop_removal.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/fog_streak_removal.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/all_weather_restore.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/deweather_cnn.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/transformer_deweather.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/frequency_deweather.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/diffusion_deweather.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/prompt_deweather.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/mamba_deweather.py`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-image-deweathering/dlhub/vision/image_deweathering_zoo.py`

**Step 1: Verify the direction does not exist yet**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.image_deweathering_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package entrypoint**

Implement `__init__.py` using the same lazy builder import pattern as existing `vision` direction packages.

Builder suffix:

```python
build_<family>_deweatherer(...)
```

**Step 3: Add the 10 family files**

Each family file must define `_VARIANTS` for `tiny`, `small`, and `base`, expose one builder, stay
toy-first and CPU-friendly, and include a `__main__` smoke path. Keep outputs weather-restoration
focused rather than dataset-specific.

**Step 4: Add `image_deweathering_zoo.py`**

Follow the explicit `_FAMILIES + _SIZES` registry pattern already used by
`dlhub/vision/image_deraining_zoo.py`.

Use prefix `deweather:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.image_deweathering.snow_removal import build_snow_removal_deweatherer as f; print(type(f(in_channels=3, variant='snow_removal_tiny')).__name__)"
python -c "from dlhub.vision.image_deweathering_zoo import list_local_arches; print(any('snow_removal_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `snow_removal_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/vision/image_deweathering dlhub/vision/image_deweathering_zoo.py
@'
Add toy-first image deweathering families as a standalone direction

This branch lands the local vision direction package and its explicit zoo
without touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 3: Add the `transparent_depth_estimation` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/glass_depth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/refract_depth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/trimap_depth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/boundary_depth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/layered_depth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/transformer_transdepth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/geometry_transdepth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/diffusion_transdepth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/prompt_transdepth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/mamba_transdepth.py`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-transparent-depth-estimation/dlhub/vision/transparent_depth_estimation_zoo.py`

**Step 1: Verify the direction currently fails to import**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.transparent_depth_estimation_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_transparent_depth_model(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and transparent-geometry focused. Do not add extra shared abstractions
unless at least three files need the same helper.

**Step 4: Add `transparent_depth_estimation_zoo.py`**

Use the same explicit registry pattern as other recent `vision` direction zoos and prefix
`transdepth:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.transparent_depth_estimation.glass_depth import build_glass_depth_transparent_depth_model as f; print(type(f(in_channels=3, variant='glass_depth_tiny')).__name__)"
python -c "from dlhub.vision.transparent_depth_estimation_zoo import list_local_arches; print(any('glass_depth_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `glass_depth_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/vision/transparent_depth_estimation dlhub/vision/transparent_depth_estimation_zoo.py
@'
Add toy-first transparent depth estimation families as a standalone direction

This branch lands the local vision direction package and its explicit zoo
without touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 4: Add the `pointcloud_forecasting` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/pointlstm_forecast.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/trajpoint_forecast.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/motion_field_forecast.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/scene_memory_forecast.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/graph_forecast3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/transformer_forecast3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/diffusion_forecast3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/prompt_forecast3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/occupancy_forecast3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/mamba_forecast3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-forecasting/dlhub/pointcloud/pointcloud_forecasting_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.pointcloud_forecasting_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_forecasting_model(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and temporally aware. Do not add extra shared abstractions unless at least
three files need the same helper.

**Step 4: Add `pointcloud_forecasting_zoo.py`**

Prefer AST-based discovery over a handwritten family list so future additions stay cheap.

Use prefix `pcforecast:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.pointcloud_forecasting.pointlstm_forecast import build_pointlstm_forecast_forecasting_model as f; print(type(f(in_channels=3, variant='pointlstm_forecast_tiny')).__name__)"
python -c "from dlhub.pointcloud.pointcloud_forecasting_zoo import list_local_arches; print(any('pointlstm_forecast_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `pointlstm_forecast_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/pointcloud/pointcloud_forecasting dlhub/pointcloud/pointcloud_forecasting_zoo.py
@'
Add toy-first point cloud forecasting families as a standalone direction

This branch lands the local pointcloud package and discovery zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 5: Add the `pointcloud_anomaly_detection` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/recon_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/patchcore_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/student_teacher_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/memory_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/density_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/transformer_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/diffusion_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/prompt_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/openvocab_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/mamba_anomaly3d.py`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-pointcloud-anomaly-detection/dlhub/pointcloud/pointcloud_anomaly_detection_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.pointcloud_anomaly_detection_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_anomaly_detector(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and anomaly-focused. Do not add extra shared abstractions unless at least
three files need the same helper.

**Step 4: Add `pointcloud_anomaly_detection_zoo.py`**

Prefer AST-based discovery similar to recent point-cloud direction zoos and use prefix
`pcanomaly:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.pointcloud_anomaly_detection.recon_anomaly3d import build_recon_anomaly3d_anomaly_detector as f; print(type(f(in_channels=3, variant='recon_anomaly3d_tiny')).__name__)"
python -c "from dlhub.pointcloud.pointcloud_anomaly_detection_zoo import list_local_arches; print(any('recon_anomaly3d_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `recon_anomaly3d_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/pointcloud/pointcloud_anomaly_detection dlhub/pointcloud/pointcloud_anomaly_detection_zoo.py
@'
Add toy-first point cloud anomaly detection families as a standalone direction

This branch lands the local pointcloud package and discovery zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 6: Add the `video_text_retrieval` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/clip4clip_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/xpool_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/frozen_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/dual_encoder_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/cross_encoder_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/temporal_align_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/transformer_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/retrieval_aug_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/prompt_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/mamba_vtr.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-video-text-retrieval/dlhub/multimodal/video_text_retrieval_zoo.py`

**Step 1: Verify the direction does not exist yet**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.video_text_retrieval_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package entrypoint**

Implement `__init__.py` using the same lazy builder import pattern as existing `multimodal`
direction packages.

Builder suffix:

```python
build_<family>_retriever(...)
```

**Step 3: Add the 10 family files**

Each family file must define `_VARIANTS` for `tiny`, `small`, and `base`, expose one builder, stay
toy-first and CPU-friendly, and include a `__main__` smoke path.

**Step 4: Add `video_text_retrieval_zoo.py`**

Follow the explicit `_FAMILIES + _SIZES` registry pattern already used by
`dlhub/multimodal/image_text_retrieval_zoo.py`.

Use prefix `vtr:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.video_text_retrieval.clip4clip_retrieval import build_clip4clip_retrieval_retriever as f; print(type(f(in_channels=3, variant='clip4clip_retrieval_tiny')).__name__)"
python -c "from dlhub.multimodal.video_text_retrieval_zoo import list_local_arches; print(any('clip4clip_retrieval_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `clip4clip_retrieval_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/multimodal/video_text_retrieval dlhub/multimodal/video_text_retrieval_zoo.py
@'
Add toy-first video-text retrieval families as a standalone direction

This branch lands the local multimodal direction package and its explicit zoo
without touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 7: Add the `embodied_question_answering` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/navqa_embodied.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/memory_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/objectnav_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/mapqa_embodied.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/speaker_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/transformer_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/grounded_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/retrieval_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/prompt_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/mamba_eqa.py`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-embodied-question-answering/dlhub/multimodal/embodied_question_answering_zoo.py`

**Step 1: Verify the direction does not exist yet**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.embodied_question_answering_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package entrypoint**

Implement `__init__.py` using the same lazy builder import pattern as existing `multimodal`
direction packages.

Builder suffix:

```python
build_<family>_embodied_qa_model(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and navigation-aware. Do not add extra shared abstractions unless at least
three files need the same helper.

**Step 4: Add `embodied_question_answering_zoo.py`**

Follow the explicit `_FAMILIES + _SIZES` registry pattern already used by recent multimodal
direction zoos.

Use prefix `eqa:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.embodied_question_answering.navqa_embodied import build_navqa_embodied_embodied_qa_model as f; print(type(f(in_channels=3, variant='navqa_embodied_tiny')).__name__)"
python -c "from dlhub.multimodal.embodied_question_answering_zoo import list_local_arches; print(any('navqa_embodied_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `navqa_embodied_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/multimodal/embodied_question_answering dlhub/multimodal/embodied_question_answering_zoo.py
@'
Add toy-first embodied question answering families as a standalone direction

This branch lands the local multimodal direction package and its explicit zoo
without touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 8: Add the `audio_text_understanding` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/audio_bert_understanding.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/wav2text_understanding.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/contrastive_atu.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/event_audio_text.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/speech_audio_text.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/transformer_atu.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/retrieval_atu.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/diffusion_atu.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/prompt_atu.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/mamba_atu.py`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-audio-text-understanding/dlhub/multimodal/audio_text_understanding_zoo.py`

**Step 1: Verify the direction does not exist yet**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.audio_text_understanding_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package entrypoint**

Implement `__init__.py` using the same lazy builder import pattern as existing `multimodal`
direction packages.

Builder suffix:

```python
build_<family>_audio_text_model(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and audio-language focused. Do not add extra shared abstractions unless at
least three files need the same helper.

**Step 4: Add `audio_text_understanding_zoo.py`**

Follow the explicit `_FAMILIES + _SIZES` registry pattern already used by recent multimodal
direction zoos.

Use prefix `atu:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.audio_text_understanding.audio_bert_understanding import build_audio_bert_understanding_audio_text_model as f; print(type(f(in_channels=3, variant='audio_bert_understanding_tiny')).__name__)"
python -c "from dlhub.multimodal.audio_text_understanding_zoo import list_local_arches; print(any('audio_bert_understanding_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `audio_bert_understanding_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/multimodal/audio_text_understanding dlhub/multimodal/audio_text_understanding_zoo.py
@'
Add toy-first audio-text understanding families as a standalone direction

This branch lands the local multimodal direction package and its explicit zoo
without touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 9: Add the `text_to_video` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/zeroscope_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/modelscope_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/lavie_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/motion_prior_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/cascade_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/control_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/transformer_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/diffusion_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/prompt_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/mamba_t2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-text-to-video/dlhub/generative/text_to_video_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.text_to_video_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_text_to_video(...)
```

**Step 3: Add the 10 family files**

Reuse the same lazy package conventions already used by `dlhub/generative/image_to_video/` and
keep the implementations toy-first rather than photorealistic.

**Step 4: Add `text_to_video_zoo.py`**

Prefer AST-based discovery over a handwritten family list so future additions stay cheap.

Use prefix `t2v:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.text_to_video.zeroscope_t2v import build_zeroscope_t2v_text_to_video as f; print(type(f(in_channels=3, variant='zeroscope_t2v_tiny')).__name__)"
python -c "from dlhub.generative.text_to_video_zoo import list_local_arches; print(any('zeroscope_t2v_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `zeroscope_t2v_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/generative/text_to_video dlhub/generative/text_to_video_zoo.py
@'
Add toy-first text-to-video families as a standalone direction

This branch lands the local generative package and discovery zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 10: Add the `video_to_video` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/vid2vid_translation.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/style_vid2vid.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/tokenflow_vid2vid.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/control_vid2vid.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/temporal_consistency_v2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/motion_transfer_v2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/transformer_v2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/diffusion_v2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/prompt_v2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/mamba_v2v.py`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-video-to-video/dlhub/generative/video_to_video_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.video_to_video_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_video_to_video(...)
```

**Step 3: Add the 10 family files**

Keep them toy-first and temporally consistent rather than photorealistic.

**Step 4: Add `video_to_video_zoo.py`**

Prefer AST-based discovery similar to the recent generative direction zoos and use prefix `v2v:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.video_to_video.vid2vid_translation import build_vid2vid_translation_video_to_video as f; print(type(f(in_channels=3, variant='vid2vid_translation_tiny')).__name__)"
python -c "from dlhub.generative.video_to_video_zoo import list_local_arches; print(any('vid2vid_translation_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `vid2vid_translation_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/generative/video_to_video dlhub/generative/video_to_video_zoo.py
@'
Add toy-first video-to-video families as a standalone direction

This branch lands the local generative package and discovery zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 11: Add the `world_models` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/rssm_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/dreamer_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/video_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/latent_dynamics_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/action_conditioned_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/memory_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/transformer_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/diffusion_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/prompt_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/mamba_world.py`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models/README.md`
- Create: `F:/DL-Hub/.worktrees/batch160-world-models/dlhub/generative/world_models_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.world_models_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_world_model(...)
```

**Step 3: Add the 10 family files**

Keep them toy-first and dynamics-focused rather than benchmark-specific.

**Step 4: Add `world_models_zoo.py`**

Prefer AST-based discovery similar to the recent generative direction zoos and use prefix `world:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.world_models.rssm_world import build_rssm_world_world_model as f; print(type(f(in_channels=3, variant='rssm_world_tiny')).__name__)"
python -c "from dlhub.generative.world_models_zoo import list_local_arches; print(any('rssm_world_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `rssm_world_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/generative/world_models dlhub/generative/world_models_zoo.py
@'
Add toy-first world-model families as a standalone direction

This branch lands the local generative package and discovery zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 12: Integrate the batch on the shared branch

**Files:**
- Modify: `F:/DL-Hub/.worktrees/batch160-integration/README.md`
- Modify: `F:/DL-Hub/.worktrees/batch160-integration/dlhub/pointcloud/__init__.py`
- Modify: `F:/DL-Hub/.worktrees/batch160-integration/dlhub/multimodal/__init__.py`
- Modify: `F:/DL-Hub/.worktrees/batch160-integration/dlhub/generative/__init__.py`

**Step 1: Merge the 10 direction branches into the integration branch without committing yet**

Run:

```powershell
git merge --no-ff --no-commit batch160-image-deweathering batch160-transparent-depth-estimation batch160-pointcloud-forecasting batch160-pointcloud-anomaly-detection batch160-video-text-retrieval batch160-embodied-question-answering batch160-audio-text-understanding batch160-text-to-video batch160-video-to-video batch160-world-models
```

Expected: merge succeeds without conflicts and leaves the staged combined result ready for shared
integration edits.

**Step 2: Wire the shared domain exports**

Update:

- `dlhub/pointcloud/__init__.py` to expose `pointcloud_forecasting_zoo.py` and
  `pointcloud_anomaly_detection_zoo.py`
- `dlhub/multimodal/__init__.py` to expose `video_text_retrieval_zoo.py`,
  `embodied_question_answering_zoo.py`, and `audio_text_understanding_zoo.py`
- `dlhub/generative/__init__.py` to expose `text_to_video_zoo.py`, `video_to_video_zoo.py`, and
  `world_models_zoo.py`

**Step 3: Append one new batch table to `README.md`**

Add exactly one new "Additional New Directions / 新增研究方向（十六）" section covering the 10 approved
directions.

**Step 4: Run final cross-domain verification**

Run:

```powershell
python -c "from dlhub.vision.image_deweathering_zoo import list_local_arches; print(any('snow_removal_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.pointcloud.pointcloud_forecasting_zoo import list_local_arches; print(any('pointlstm_forecast_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.multimodal.video_text_retrieval_zoo import list_local_arches; print(any('clip4clip_retrieval_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.generative.text_to_video_zoo import list_local_arches; print(any('zeroscope_t2v_tiny' in x for x in list_local_arches()))"
git diff --check
```

Expected: each representative direction is discoverable and `git diff --check` reports no patch
formatting errors.

**Step 5: Commit the integration branch with a Lore message**

Run:

```powershell
git add README.md dlhub/pointcloud/__init__.py dlhub/multimodal/__init__.py dlhub/generative/__init__.py
@'
Integrate the sixteenth cross-domain direction batch into the shared surfaces

This commit merges the ten direction branches, wires the shared exports, and
records the new batch in the top-level README while keeping the rollout
append-only.

Constraint: User approved package-only expansion with minimal verification
Confidence: high
Scope-risk: moderate
Directive: Keep future batch recording append-only in README and shared exports
Tested: Direction smoke checks, cross-domain smoke, git diff --check
Not-tested: Pytest or lesson coverage
'@ | git commit -F -
```

**Step 6: Verify the integration branch is clean before merge-back**

Run:

```powershell
git status --short --branch
```

Expected: only the branch header is printed, with no remaining tracked modifications.

**Step 7: Merge the integration branch back to `main` with a Lore message**

Run:

```powershell
git checkout main
git merge --no-ff --no-commit batch160-integration
@'
Expand the zoo with a sixteenth cross-domain 100-family batch

This merge lands ten new standalone directions across vision, pointcloud,
multimodal, and generative while preserving the package-only, minimal-smoke
contract approved for this batch.

Constraint: The batch must avoid lesson work and new pytest files
Rejected: Merge direction branches directly into main | integration branch keeps shared wiring isolated
Confidence: high
Scope-risk: moderate
Directive: Keep batch expansions append-only and land shared wiring only after direction smokes pass
Tested: Direction smoke checks, cross-domain smoke, git diff --check
Not-tested: Pytest or lesson coverage
'@ | git commit -F -
```

Expected: `main` now contains the full batch and the merge commit follows the Lore protocol.

**Step 8: Remove the worktrees after the merge**

Run:

```powershell
git worktree remove .worktrees/batch160-image-deweathering
git worktree remove .worktrees/batch160-transparent-depth-estimation
git worktree remove .worktrees/batch160-pointcloud-forecasting
git worktree remove .worktrees/batch160-pointcloud-anomaly-detection
git worktree remove .worktrees/batch160-video-text-retrieval
git worktree remove .worktrees/batch160-embodied-question-answering
git worktree remove .worktrees/batch160-audio-text-understanding
git worktree remove .worktrees/batch160-text-to-video
git worktree remove .worktrees/batch160-video-to-video
git worktree remove .worktrees/batch160-world-models
git worktree remove .worktrees/batch160-integration
```

Expected: only the main worktree remains.
