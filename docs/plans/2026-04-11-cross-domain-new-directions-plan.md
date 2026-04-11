# Cross-Domain New Directions Batch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 100 new toy-first algorithm families across 10 previously unimplemented directions spanning `vision`, `pointcloud`, `multimodal`, and `generative`, without adding lessons or pytest files.

**Architecture:** Create one package per new direction, one family per file, and one direction-local zoo or registry surface per direction where that pattern is already established. Keep direction worktrees local-first and reserve shared `README.md` plus domain `__init__.py` wiring for one final integration branch.

**Tech Stack:** Python 3, `torch`, repo-local toy model helpers, lazy package `__init__.py` files, direction-specific `*_zoo.py` modules, `.worktrees/` git worktrees, and minimal import/list smoke verification.

---

### Task 1: Prepare the worktree lane layout

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/`
- Create: `F:/DL-Hub/.worktrees/batch140-integration/`

**Step 1: Verify the worktree root is ignored**

Run:

```powershell
git check-ignore -q .worktrees
```

Expected: exit code `0`.

**Step 2: Create the 10 direction worktrees plus the integration worktree**

Run:

```powershell
git worktree add .worktrees/batch140-image-relighting -b batch140-image-relighting
git worktree add .worktrees/batch140-transparent-object-segmentation -b batch140-transparent-object-segmentation
git worktree add .worktrees/batch140-video-matting -b batch140-video-matting
git worktree add .worktrees/batch140-event-camera-understanding -b batch140-event-camera-understanding
git worktree add .worktrees/batch140-scene-flow -b batch140-scene-flow
git worktree add .worktrees/batch140-pointcloud-completion -b batch140-pointcloud-completion
git worktree add .worktrees/batch140-audio-visual-learning -b batch140-audio-visual-learning
git worktree add .worktrees/batch140-multimodal-reasoning -b batch140-multimodal-reasoning
git worktree add .worktrees/batch140-video-diffusion -b batch140-video-diffusion
git worktree add .worktrees/batch140-text-to-3d -b batch140-text-to-3d
git worktree add .worktrees/batch140-integration -b batch140-integration
```

Expected: each worktree is created from the same starting commit.

**Step 3: Capture the clean baseline**

Run:

```powershell
git -C .worktrees/batch140-image-relighting status --short --branch
git -C .worktrees/batch140-integration status --short --branch
```

Expected: branch shown, no tracked modifications.

### Task 2: Add the `image_relighting` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/deep_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/hdr_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/intrinsic_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/ratio_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/retinex_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/portrait_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/transformer_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/diffusion_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/prompt_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/mamba_relight.py`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-image-relighting/dlhub/vision/image_relighting_zoo.py`

**Step 1: Verify the direction does not exist yet**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.image_relighting_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package entrypoint**

Implement `__init__.py` using the same lazy builder import pattern as existing `vision` direction packages.

Builder suffix:

```python
build_<family>_relighter(...)
```

**Step 3: Add the 10 family files**

Each family file must define `_VARIANTS` for `tiny`, `small`, and `base`, expose one builder, stay
toy-first and CPU-friendly, and include a `__main__` smoke path.

**Step 4: Add `image_relighting_zoo.py`**

Follow the explicit `_FAMILIES` + `_SIZES` registry pattern already used by
`dlhub/vision/video_frame_interpolation_zoo.py`.

Use prefix `relight:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.image_relighting.deep_relight import build_deep_relight_relighter as f; print(type(f(in_channels=3, variant='deep_relight_tiny')).__name__)"
python -c "from dlhub.vision.image_relighting_zoo import list_local_arches; print(any('deep_relight_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `deep_relight_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/vision/image_relighting dlhub/vision/image_relighting_zoo.py
git commit -m "Add image relighting toy families"
```

### Task 3: Add the `transparent_object_segmentation` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/glassseg_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/translab_seg.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/refractmask_seg.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/camotransparent_seg.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/trimap_transparent.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/boundary_glass_seg.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/transformer_transparent.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/diffusion_transparent.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/prompt_transparent.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/mamba_transparent.py`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-transparent-object-segmentation/dlhub/vision/transparent_object_segmentation_zoo.py`

**Step 1: Verify the direction currently fails to import**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.transparent_object_segmentation_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_transparent_segmenter(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and segmentation-oriented. Do not add extra shared abstractions unless at
least three files need the same helper.

**Step 4: Add `transparent_object_segmentation_zoo.py`**

Use the same explicit registry pattern as other recent `vision` direction zoos and prefix `tos:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.transparent_object_segmentation.glassseg_toy import build_glassseg_toy_transparent_segmenter as f; print(type(f(in_channels=3, variant='glassseg_toy_tiny')).__name__)"
python -c "from dlhub.vision.transparent_object_segmentation_zoo import list_local_arches; print(any('glassseg_toy_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `glassseg_toy_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/vision/transparent_object_segmentation dlhub/vision/transparent_object_segmentation_zoo.py
git commit -m "Add transparent object segmentation toy families"
```

### Task 4: Add the `video_matting` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/dim_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/fba_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/rvm_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/gca_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/tcvomatting.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/memory_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/transformer_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/flow_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/prompt_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/mamba_vmatte.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-video-matting/dlhub/vision/video_matting_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.video_matting_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_video_matter(...)
```

**Step 3: Add the 10 family files**

Use toy-first video matting outputs and keep temporal handling lightweight.

**Step 4: Add `video_matting_zoo.py`**

Use prefix `vmatte:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.video_matting.dim_vmatte import build_dim_vmatte_video_matter as f; print(type(f(in_channels=3, variant='dim_vmatte_tiny')).__name__)"
python -c "from dlhub.vision.video_matting_zoo import list_local_arches; print(any('dim_vmatte_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `dim_vmatte_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/vision/video_matting dlhub/vision/video_matting_zoo.py
git commit -m "Add video matting toy families"
```

### Task 5: Add the `event_camera_understanding` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/ev_cnn.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/voxel_eventnet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/spike_eventnet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/event_unet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/event_tracker.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/event_depth.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/transformer_event.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/state_space_event.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/crossmodal_event.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/mamba_event.py`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-event-camera-understanding/dlhub/vision/event_camera_understanding_zoo.py`

**Step 1: Confirm the zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.event_camera_understanding_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_event_model(...)
```

**Step 3: Add the 10 family files**

Keep the event tensor contracts simple and CPU-friendly.

**Step 4: Add `event_camera_understanding_zoo.py`**

Use the same explicit registry pattern as other recent `vision` direction zoos and prefix `event:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.event_camera_understanding.ev_cnn import build_ev_cnn_event_model as f; print(type(f(in_channels=3, variant='ev_cnn_tiny')).__name__)"
python -c "from dlhub.vision.event_camera_understanding_zoo import list_local_arches; print(any('ev_cnn_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `ev_cnn_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/vision/event_camera_understanding dlhub/vision/event_camera_understanding_zoo.py
git commit -m "Add event camera understanding toy families"
```

### Task 6: Add the `scene_flow` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/flow3d_pointnet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/flow3d_flownet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/pointpwc_flow.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/raft3d_points.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/iter_flow3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/cost_volume_flow3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/transformer_flow3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/diffusion_flow3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/prompt_flow3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/mamba_flow3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-scene-flow/dlhub/pointcloud/scene_flow_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.scene_flow_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_scene_flow_estimator(...)
```

**Step 3: Add the 10 family files**

Stay toy-first and follow the pointcloud direction-package style already used by `gaussian_splatting`
and `tracking3d`.

**Step 4: Add `scene_flow_zoo.py`**

Use the AST-discovery style already used by `dlhub/pointcloud/tracking3d_zoo.py` and
`dlhub/pointcloud/gaussian_splatting_zoo.py`, with prefix `pcsf:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.scene_flow.flow3d_pointnet import build_flow3d_pointnet_scene_flow_estimator as f; print(type(f(in_channels=3, variant='flow3d_pointnet_tiny')).__name__)"
python -c "from dlhub.pointcloud.scene_flow_zoo import list_local_arches; print(any('flow3d_pointnet_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `flow3d_pointnet_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/pointcloud/scene_flow dlhub/pointcloud/scene_flow_zoo.py
git commit -m "Add scene flow toy families"
```

### Task 7: Add the `pointcloud_completion` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/pcn_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/topnet_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/grnet_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/snowflake_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/folding_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/anchor_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/transformer_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/diffusion_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/text_guided_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/mamba_completion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-pointcloud-completion/dlhub/pointcloud/pointcloud_completion_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.pointcloud_completion_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_completer(...)
```

**Step 3: Add the 10 family files**

Keep them toy-first and pointcloud-completion specific.

**Step 4: Add `pointcloud_completion_zoo.py`**

Use the same AST-discovery registry pattern as other pointcloud direction zoos and prefix `pccomp:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.pointcloud_completion.pcn_completion import build_pcn_completion_completer as f; print(type(f(in_channels=3, variant='pcn_completion_tiny')).__name__)"
python -c "from dlhub.pointcloud.pointcloud_completion_zoo import list_local_arches; print(any('pcn_completion_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `pcn_completion_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/pointcloud/pointcloud_completion dlhub/pointcloud/pointcloud_completion_zoo.py
git commit -m "Add pointcloud completion toy families"
```

### Task 8: Add the `audio_visual_learning` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/av_syncnet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/av_contrast.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/av_fusionnet.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/av_localizer.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/av_separation.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/av_caption_bridge.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/transformer_av.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/diffusion_av.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/prompt_av.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/mamba_av.py`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-audio-visual-learning/dlhub/multimodal/audio_visual_learning_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.audio_visual_learning_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_audio_visual_model(...)
```

**Step 3: Add the 10 family files**

Use the lazy package style already present under `dlhub/multimodal/vlm/`.

**Step 4: Add `audio_visual_learning_zoo.py`**

Follow the explicit `_FAMILIES` + `_SIZES` multimodal zoo style already used by
`dlhub/multimodal/prompt_learning_zoo.py`, with prefix `avl:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.audio_visual_learning.av_syncnet import build_av_syncnet_audio_visual_model as f; print(type(f(in_channels=3, variant='av_syncnet_tiny')).__name__)"
python -c "from dlhub.multimodal.audio_visual_learning_zoo import list_local_arches; print(any('av_syncnet_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `av_syncnet_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/multimodal/audio_visual_learning dlhub/multimodal/audio_visual_learning_zoo.py
git commit -m "Add audio-visual learning toy families"
```

### Task 9: Add the `multimodal_reasoning` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/vqa_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/chain_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/tool_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/memory_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/grounded_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/program_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/transformer_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/retrieval_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/prompt_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/mamba_reasoner.py`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-multimodal-reasoning/dlhub/multimodal/multimodal_reasoning_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.multimodal_reasoning_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_reasoner(...)
```

**Step 3: Add the 10 family files**

Keep the outputs toy-first and reasoning-oriented without introducing tool backends or external APIs.

**Step 4: Add `multimodal_reasoning_zoo.py`**

Use the same explicit multimodal zoo pattern and prefix `mmr:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.multimodal_reasoning.vqa_reasoner import build_vqa_reasoner_reasoner as f; print(type(f(in_channels=3, variant='vqa_reasoner_tiny')).__name__)"
python -c "from dlhub.multimodal.multimodal_reasoning_zoo import list_local_arches; print(any('vqa_reasoner_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `vqa_reasoner_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/multimodal/multimodal_reasoning dlhub/multimodal/multimodal_reasoning_zoo.py
git commit -m "Add multimodal reasoning toy families"
```

### Task 10: Add the `video_diffusion` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/latent_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/frame_interp_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/video_unet_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/cascade_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/motion_prior_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/control_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/transformer_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/rectified_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/prompt_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/mamba_video_diffusion.py`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-video-diffusion/dlhub/generative/video_diffusion_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.video_diffusion_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_video_diffusion(...)
```

**Step 3: Add the 10 family files**

Reuse the same lazy package conventions already used by `dlhub/generative/diffusion/`.

**Step 4: Add `video_diffusion_zoo.py`**

Prefer AST-based discovery over a hand-written family list so future additions stay cheap. Use prefix `vdiff:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.video_diffusion.latent_video_diffusion import build_latent_video_diffusion_video_diffusion as f; print(type(f(in_channels=3, variant='latent_video_diffusion_tiny')).__name__)"
python -c "from dlhub.generative.video_diffusion_zoo import list_local_arches; print(any('latent_video_diffusion_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `latent_video_diffusion_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/generative/video_diffusion dlhub/generative/video_diffusion_zoo.py
git commit -m "Add video diffusion toy families"
```

### Task 11: Add the `text_to_3d` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/dreamfusion_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/magic3d_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/score_distill_3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/neural_lift_3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/sdf_prompt_3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/mesh_diffuse_3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/transformer_text3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/gaussian_text3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/layout_text3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/mamba_text3d.py`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d/README.md`
- Create: `F:/DL-Hub/.worktrees/batch140-text-to-3d/dlhub/generative/text_to_3d_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.text_to_3d_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_text3d_generator(...)
```

**Step 3: Add the 10 family files**

Keep them toy-first and structural rather than photorealistic.

**Step 4: Add `text_to_3d_zoo.py`**

Prefer AST-based discovery similar to the generative diffusion zoo and use prefix `txt3d:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.text_to_3d.dreamfusion_toy import build_dreamfusion_toy_text3d_generator as f; print(type(f(in_channels=3, variant='dreamfusion_toy_tiny')).__name__)"
python -c "from dlhub.generative.text_to_3d_zoo import list_local_arches; print(any('dreamfusion_toy_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `dreamfusion_toy_tiny`.

**Step 6: Commit the worktree branch**

Run:

```powershell
git add dlhub/generative/text_to_3d dlhub/generative/text_to_3d_zoo.py
git commit -m "Add text-to-3D toy families"
```

### Task 12: Integrate the batch on the shared branch

**Files:**
- Modify: `F:/DL-Hub/.worktrees/batch140-integration/README.md`
- Modify: `F:/DL-Hub/.worktrees/batch140-integration/dlhub/pointcloud/__init__.py`
- Modify: `F:/DL-Hub/.worktrees/batch140-integration/dlhub/multimodal/__init__.py`
- Modify: `F:/DL-Hub/.worktrees/batch140-integration/dlhub/generative/__init__.py`

**Step 1: Merge the 10 direction branches into the integration branch**

Run:

```powershell
git merge --no-ff batch140-image-relighting
git merge --no-ff batch140-transparent-object-segmentation
git merge --no-ff batch140-video-matting
git merge --no-ff batch140-event-camera-understanding
git merge --no-ff batch140-scene-flow
git merge --no-ff batch140-pointcloud-completion
git merge --no-ff batch140-audio-visual-learning
git merge --no-ff batch140-multimodal-reasoning
git merge --no-ff batch140-video-diffusion
git merge --no-ff batch140-text-to-3d
```

Expected: all merges succeed without touching unrelated files.

**Step 2: Wire the shared domain exports**

Update:

- `dlhub/pointcloud/__init__.py` to expose `scene_flow_zoo.py` and `pointcloud_completion_zoo.py`
- `dlhub/multimodal/__init__.py` to expose `audio_visual_learning_zoo.py` and `multimodal_reasoning_zoo.py`
- `dlhub/generative/__init__.py` to expose `video_diffusion_zoo.py` and `text_to_3d_zoo.py`

**Step 3: Append one new batch table to `README.md`**

Add exactly one new "Additional New Directions" section covering the 10 approved directions.

**Step 4: Run final cross-domain verification**

Run:

```powershell
python -c "from dlhub.vision.image_relighting_zoo import list_local_arches; print(any('deep_relight_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.pointcloud.scene_flow_zoo import list_local_arches; print(any('flow3d_pointnet_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.multimodal.audio_visual_learning_zoo import list_local_arches; print(any('av_syncnet_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.generative.video_diffusion_zoo import list_local_arches; print(any('latent_video_diffusion_tiny' in x for x in list_local_arches()))"
git diff --check
```

Expected: each representative direction is discoverable and `git diff --check` reports no patch formatting errors.

**Step 5: Merge the integration branch back to `main`**

Run:

```powershell
git checkout main
git merge --no-ff batch140-integration
```

Use a Lore-format merge commit message describing why this 100-family batch was added.

**Step 6: Remove the worktrees after the merge**

Run:

```powershell
git worktree remove .worktrees/batch140-image-relighting
git worktree remove .worktrees/batch140-transparent-object-segmentation
git worktree remove .worktrees/batch140-video-matting
git worktree remove .worktrees/batch140-event-camera-understanding
git worktree remove .worktrees/batch140-scene-flow
git worktree remove .worktrees/batch140-pointcloud-completion
git worktree remove .worktrees/batch140-audio-visual-learning
git worktree remove .worktrees/batch140-multimodal-reasoning
git worktree remove .worktrees/batch140-video-diffusion
git worktree remove .worktrees/batch140-text-to-3d
git worktree remove .worktrees/batch140-integration
```

Expected: only the main worktree remains.
