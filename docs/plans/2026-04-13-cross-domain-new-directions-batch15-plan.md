# Cross-Domain New Directions Batch 15 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add 100 new toy-first algorithm families across 10 previously unimplemented directions spanning `vision`, `pointcloud`, `multimodal`, and `generative`, without adding lessons, dependencies, or pytest files.

**Architecture:** Create one package per new direction, one family per file, and one direction-local zoo or registry surface per direction. Keep each direction branch local-only, then land shared README and domain export wiring on one integration branch after all direction-level smoke checks pass.

**Tech Stack:** Python 3, `torch`, repo-local toy model helpers, lazy package `__init__.py` files, explicit registries for `vision` and `multimodal`, AST-discovery zoos for `pointcloud` and `generative`, `.worktrees/` git worktrees, and minimal import/list smoke verification.

---

### Task 1: Prepare the worktree lane layout

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/`
- Create: `F:/DL-Hub/.worktrees/batch150-integration/`

**Step 1: Verify the worktree root is ignored**

Run:

```powershell
git check-ignore -q .worktrees
```

Expected: exit code `0`.

**Step 2: Create the 10 direction worktrees plus the integration worktree**

Run:

```powershell
git worktree add .worktrees/batch150-image-deraining -b batch150-image-deraining
git worktree add .worktrees/batch150-shadow-detection -b batch150-shadow-detection
git worktree add .worktrees/batch150-pointcloud-upsampling -b batch150-pointcloud-upsampling
git worktree add .worktrees/batch150-shape-correspondence-3d -b batch150-shape-correspondence-3d
git worktree add .worktrees/batch150-open-vocabulary-3d -b batch150-open-vocabulary-3d
git worktree add .worktrees/batch150-image-text-retrieval -b batch150-image-text-retrieval
git worktree add .worktrees/batch150-vision-language-navigation -b batch150-vision-language-navigation
git worktree add .worktrees/batch150-document-vlm -b batch150-document-vlm
git worktree add .worktrees/batch150-image-to-video -b batch150-image-to-video
git worktree add .worktrees/batch150-image-to-3d -b batch150-image-to-3d
git worktree add .worktrees/batch150-integration -b batch150-integration
```

Expected: each worktree is created from the same starting commit.

**Step 3: Capture the clean baseline**

Run:

```powershell
git -C .worktrees/batch150-image-deraining status --short --branch
git -C .worktrees/batch150-integration status --short --branch
```

Expected: each branch is shown with no tracked modifications.

### Task 2: Add the `image_deraining` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/jorder_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/did_mdn_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/resguide_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/recurrent_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/density_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/transformer_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/frequency_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/diffusion_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/prompt_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/mamba_derain.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-image-deraining/dlhub/vision/image_deraining_zoo.py`

**Step 1: Verify the direction does not exist yet**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.image_deraining_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package entrypoint**

Implement `__init__.py` using the same lazy builder import pattern as existing `vision` direction packages.

Builder suffix:

```python
build_<family>_derainer(...)
```

**Step 3: Add the 10 family files**

Each family file must define `_VARIANTS` for `tiny`, `small`, and `base`, expose one builder, stay
toy-first and CPU-friendly, and include a `__main__` smoke path.

**Step 4: Add `image_deraining_zoo.py`**

Follow the explicit `_FAMILIES` + `_SIZES` registry pattern already used by
`dlhub/vision/video_frame_interpolation_zoo.py`.

Use prefix `derain:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.image_deraining.jorder_derain import build_jorder_derain_derainer as f; print(type(f(in_channels=3, variant='jorder_derain_tiny')).__name__)"
python -c "from dlhub.vision.image_deraining_zoo import list_local_arches; print(any('jorder_derain_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `jorder_derain_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/vision/image_deraining dlhub/vision/image_deraining_zoo.py
@'
Add toy-first image deraining families as a standalone direction

This branch lands the local direction package and its explicit zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 3: Add the `shadow_detection` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/dsd_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/bdrar_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/stacked_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/context_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/boundary_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/transformer_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/diffusion_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/prompt_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/state_space_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/mamba_shadow.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-shadow-detection/dlhub/vision/shadow_detection_zoo.py`

**Step 1: Verify the direction currently fails to import**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.vision.shadow_detection_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_shadow_detector(...)
```

**Step 3: Add the 10 family files**

Keep outputs toy-first and shadow-focused. Do not add extra shared abstractions unless at least
three files need the same helper.

**Step 4: Add `shadow_detection_zoo.py`**

Use the same explicit registry pattern as other recent `vision` direction zoos and prefix `shadow:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.vision.shadow_detection.dsd_shadow import build_dsd_shadow_shadow_detector as f; print(type(f(in_channels=3, variant='dsd_shadow_tiny')).__name__)"
python -c "from dlhub.vision.shadow_detection_zoo import list_local_arches; print(any('dsd_shadow_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `dsd_shadow_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/vision/shadow_detection dlhub/vision/shadow_detection_zoo.py
@'
Add toy-first shadow detection families as a standalone direction

This branch lands the local direction package and its explicit zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 4: Add the `pointcloud_upsampling` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/punet_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/mpu_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/pugan_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/dispu_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/patch_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/folding_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/transformer_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/diffusion_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/prompt_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/mamba_upsample.py`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-pointcloud-upsampling/dlhub/pointcloud/pointcloud_upsampling_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.pointcloud_upsampling_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_upsampler(...)
```

**Step 3: Add the 10 family files**

Stay toy-first and follow the pointcloud direction-package style already used by `gaussian_splatting`
and `scene_flow`.

**Step 4: Add `pointcloud_upsampling_zoo.py`**

Use the AST-discovery style already used by `dlhub/pointcloud/gaussian_splatting_zoo.py`.

Use prefix `pcup:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.pointcloud_upsampling.punet_upsample import build_punet_upsample_upsampler as f; print(type(f(in_channels=3, variant='punet_upsample_tiny')).__name__)"
python -c "from dlhub.pointcloud.pointcloud_upsampling_zoo import list_local_arches; print(any('punet_upsample_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `punet_upsample_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/pointcloud/pointcloud_upsampling dlhub/pointcloud/pointcloud_upsampling_zoo.py
@'
Add toy-first point cloud upsampling families as a standalone direction

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

### Task 5: Add the `shape_correspondence_3d` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/fmnet_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/geodesic_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/descriptor_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/deformation_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/graphmatch_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/cycle_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/transformer_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/diffusion_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/prompt_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/mamba_corr3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-shape-correspondence-3d/dlhub/pointcloud/shape_correspondence_3d_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.shape_correspondence_3d_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_shape_correspondence_model(...)
```

**Step 3: Add the 10 family files**

Stay toy-first and follow the pointcloud direction-package style already used by `gaussian_splatting`
and `scene_flow`.

**Step 4: Add `shape_correspondence_3d_zoo.py`**

Use the AST-discovery style already used by `dlhub/pointcloud/scene_flow_zoo.py`.

Use prefix `s3corr:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.shape_correspondence_3d.fmnet_corr3d import build_fmnet_corr3d_shape_correspondence_model as f; print(type(f(in_channels=3, variant='fmnet_corr3d_tiny')).__name__)"
python -c "from dlhub.pointcloud.shape_correspondence_3d_zoo import list_local_arches; print(any('fmnet_corr3d_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `fmnet_corr3d_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/pointcloud/shape_correspondence_3d dlhub/pointcloud/shape_correspondence_3d_zoo.py
@'
Add toy-first 3D shape correspondence families as a standalone direction

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

### Task 6: Add the `open_vocabulary_3d` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/clip3d_openvocab.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/regionclip3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/grounding3d_openvocab.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/retrieval3d_openvocab.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/proposal3d_openvocab.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/languagefield_3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/transformer_openvocab3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/diffusion_openvocab3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/prompt_openvocab3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/mamba_openvocab3d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-open-vocabulary-3d/dlhub/pointcloud/open_vocabulary_3d_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.pointcloud.open_vocabulary_3d_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_open_vocabulary_3d_model(...)
```

**Step 3: Add the 10 family files**

Keep the outputs toy-first and open-vocabulary oriented without introducing external APIs or remote
text backends.

**Step 4: Add `open_vocabulary_3d_zoo.py`**

Use the same AST-discovery registry pattern as other pointcloud direction zoos and prefix `ov3d:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.pointcloud.open_vocabulary_3d.clip3d_openvocab import build_clip3d_openvocab_open_vocabulary_3d_model as f; print(type(f(in_channels=3, variant='clip3d_openvocab_tiny')).__name__)"
python -c "from dlhub.pointcloud.open_vocabulary_3d_zoo import list_local_arches; print(any('clip3d_openvocab_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `clip3d_openvocab_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/pointcloud/open_vocabulary_3d dlhub/pointcloud/open_vocabulary_3d_zoo.py
@'
Add toy-first open-vocabulary 3D families as a standalone direction

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

### Task 7: Add the `image_text_retrieval` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/clip_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/albef_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/blip_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/dual_encoder_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/cross_attention_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/region_text_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/transformer_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/prompt_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/diffusion_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/mamba_retrieval.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-image-text-retrieval/dlhub/multimodal/image_text_retrieval_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.image_text_retrieval_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_retriever(...)
```

**Step 3: Add the 10 family files**

Use the lazy package style already present under `dlhub/multimodal/prompt_learning/`.

**Step 4: Add `image_text_retrieval_zoo.py`**

Follow the explicit `_FAMILIES` + `_SIZES` multimodal zoo style already used by
`dlhub/multimodal/prompt_learning_zoo.py`, with prefix `itr:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.image_text_retrieval.clip_retrieval import build_clip_retrieval_retriever as f; print(type(f(in_channels=3, variant='clip_retrieval_tiny')).__name__)"
python -c "from dlhub.multimodal.image_text_retrieval_zoo import list_local_arches; print(any('clip_retrieval_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `clip_retrieval_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/multimodal/image_text_retrieval dlhub/multimodal/image_text_retrieval_zoo.py
@'
Add toy-first image-text retrieval families as a standalone direction

This branch lands the local multimodal package and explicit zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 8: Add the `vision_language_navigation` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/seq2seq_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/speaker_follower_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/map_memory_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/object_goal_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/panoramic_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/transformer_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/grounding_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/prompt_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/diffusion_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/mamba_nav.py`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-vision-language-navigation/dlhub/multimodal/vision_language_navigation_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.vision_language_navigation_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_navigator(...)
```

**Step 3: Add the 10 family files**

Keep the outputs toy-first and navigation-oriented without introducing simulator bindings or
external environments.

**Step 4: Add `vision_language_navigation_zoo.py`**

Use the same explicit multimodal zoo pattern and prefix `vln:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.vision_language_navigation.seq2seq_nav import build_seq2seq_nav_navigator as f; print(type(f(in_channels=3, variant='seq2seq_nav_tiny')).__name__)"
python -c "from dlhub.multimodal.vision_language_navigation_zoo import list_local_arches; print(any('seq2seq_nav_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `seq2seq_nav_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/multimodal/vision_language_navigation dlhub/multimodal/vision_language_navigation_zoo.py
@'
Add toy-first vision-language navigation families as a standalone direction

This branch lands the local multimodal package and explicit zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 9: Add the `document_vlm` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/layoutlm_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/docformer_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/pix2struct_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/donut_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/table_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/chart_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/transformer_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/retrieval_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/prompt_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/mamba_doc.py`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-document-vlm/dlhub/multimodal/document_vlm_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.multimodal.document_vlm_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_document_vlm(...)
```

**Step 3: Add the 10 family files**

Keep the outputs toy-first and document-oriented without introducing OCR services or remote APIs.

**Step 4: Add `document_vlm_zoo.py`**

Use the same explicit multimodal zoo pattern and prefix `docvlm:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.multimodal.document_vlm.layoutlm_doc import build_layoutlm_doc_document_vlm as f; print(type(f(in_channels=3, variant='layoutlm_doc_tiny')).__name__)"
python -c "from dlhub.multimodal.document_vlm_zoo import list_local_arches; print(any('layoutlm_doc_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `layoutlm_doc_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/multimodal/document_vlm dlhub/multimodal/document_vlm_zoo.py
@'
Add toy-first document VLM families as a standalone direction

This branch lands the local multimodal package and explicit zoo without
touching shared exports or the global README.

Constraint: This batch must stay package-only and avoid pytest additions
Confidence: high
Scope-risk: narrow
Directive: Keep shared wiring on the integration branch
Tested: Direction builder smoke and zoo listing smoke
Not-tested: Cross-domain integration
'@ | git commit -F -
```

### Task 10: Add the `image_to_video` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/i2vgen_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/lavie_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/dynami_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/motion_adapter_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/temporal_unet_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/cascade_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/control_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/transformer_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/diffusion_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/mamba_i2v.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-video/dlhub/generative/image_to_video_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.image_to_video_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_image_to_video(...)
```

**Step 3: Add the 10 family files**

Reuse the same lazy package conventions already used by `dlhub/generative/video_diffusion/`.

**Step 4: Add `image_to_video_zoo.py`**

Prefer AST-based discovery over a handwritten family list so future additions stay cheap.

Use prefix `i2v:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.image_to_video.i2vgen_toy import build_i2vgen_toy_image_to_video as f; print(type(f(in_channels=3, variant='i2vgen_toy_tiny')).__name__)"
python -c "from dlhub.generative.image_to_video_zoo import list_local_arches; print(any('i2vgen_toy_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `i2vgen_toy_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/generative/image_to_video dlhub/generative/image_to_video_zoo.py
@'
Add toy-first image-to-video families as a standalone direction

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

### Task 11: Add the `image_to_3d` direction

**Files:**
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/__init__.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/zero123_toy.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/triplane_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/lift3d_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/mesh_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/gaussian_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/sdf_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/transformer_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/diffusion_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/prompt_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/mamba_i23d.py`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d/README.md`
- Create: `F:/DL-Hub/.worktrees/batch150-image-to-3d/dlhub/generative/image_to_3d_zoo.py`

**Step 1: Confirm the new zoo is absent**

Run:

```powershell
python -c "import importlib; importlib.import_module('dlhub.generative.image_to_3d_zoo')"
```

Expected: FAIL with `ModuleNotFoundError`.

**Step 2: Create the lazy package**

Builder suffix:

```python
build_<family>_image_to_3d_generator(...)
```

**Step 3: Add the 10 family files**

Keep them toy-first and structural rather than photorealistic.

**Step 4: Add `image_to_3d_zoo.py`**

Prefer AST-based discovery similar to the generative diffusion zoo and use prefix `i23d:`.

**Step 5: Run minimal verification**

Run:

```powershell
python -c "from dlhub.generative.image_to_3d.zero123_toy import build_zero123_toy_image_to_3d_generator as f; print(type(f(in_channels=3, variant='zero123_toy_tiny')).__name__)"
python -c "from dlhub.generative.image_to_3d_zoo import list_local_arches; print(any('zero123_toy_tiny' in x for x in list_local_arches()))"
```

Expected: import/build succeeds and zoo listing contains `zero123_toy_tiny`.

**Step 6: Commit the worktree branch with a Lore message**

Run:

```powershell
git add dlhub/generative/image_to_3d dlhub/generative/image_to_3d_zoo.py
@'
Add toy-first image-to-3D families as a standalone direction

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
- Modify: `F:/DL-Hub/.worktrees/batch150-integration/README.md`
- Modify: `F:/DL-Hub/.worktrees/batch150-integration/dlhub/pointcloud/__init__.py`
- Modify: `F:/DL-Hub/.worktrees/batch150-integration/dlhub/multimodal/__init__.py`
- Modify: `F:/DL-Hub/.worktrees/batch150-integration/dlhub/generative/__init__.py`

**Step 1: Merge the 10 direction branches into the integration branch without committing yet**

Run:

```powershell
git merge --no-ff --no-commit batch150-image-deraining batch150-shadow-detection batch150-pointcloud-upsampling batch150-shape-correspondence-3d batch150-open-vocabulary-3d batch150-image-text-retrieval batch150-vision-language-navigation batch150-document-vlm batch150-image-to-video batch150-image-to-3d
```

Expected: merge succeeds without conflicts and leaves the staged combined result ready for shared
integration edits.

**Step 2: Wire the shared domain exports**

Update:

- `dlhub/pointcloud/__init__.py` to expose `pointcloud_upsampling_zoo.py`,
  `shape_correspondence_3d_zoo.py`, and `open_vocabulary_3d_zoo.py`
- `dlhub/multimodal/__init__.py` to expose `image_text_retrieval_zoo.py`,
  `vision_language_navigation_zoo.py`, and `document_vlm_zoo.py`
- `dlhub/generative/__init__.py` to expose `image_to_video_zoo.py` and `image_to_3d_zoo.py`

**Step 3: Append one new batch table to `README.md`**

Add exactly one new "Additional New Directions / 新增研究方向（十五）" section covering the 10 approved
directions.

**Step 4: Run final cross-domain verification**

Run:

```powershell
python -c "from dlhub.vision.image_deraining_zoo import list_local_arches; print(any('jorder_derain_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.pointcloud.pointcloud_upsampling_zoo import list_local_arches; print(any('punet_upsample_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.multimodal.image_text_retrieval_zoo import list_local_arches; print(any('clip_retrieval_tiny' in x for x in list_local_arches()))"
python -c "from dlhub.generative.image_to_video_zoo import list_local_arches; print(any('i2vgen_toy_tiny' in x for x in list_local_arches()))"
git diff --check
```

Expected: each representative direction is discoverable and `git diff --check` reports no patch
formatting errors.

**Step 5: Commit the integration branch with a Lore message**

Run:

```powershell
git add README.md dlhub/pointcloud/__init__.py dlhub/multimodal/__init__.py dlhub/generative/__init__.py
@'
Integrate the fifteenth cross-domain direction batch into the shared surfaces

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

**Step 6: Merge the integration branch back to `main` with a Lore message**

Run:

```powershell
git checkout main
git merge --no-ff --no-commit batch150-integration
@'
Expand the zoo with a fifteenth cross-domain 100-family batch

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

**Step 7: Remove the worktrees after the merge**

Run:

```powershell
git worktree remove .worktrees/batch150-image-deraining
git worktree remove .worktrees/batch150-shadow-detection
git worktree remove .worktrees/batch150-pointcloud-upsampling
git worktree remove .worktrees/batch150-shape-correspondence-3d
git worktree remove .worktrees/batch150-open-vocabulary-3d
git worktree remove .worktrees/batch150-image-text-retrieval
git worktree remove .worktrees/batch150-vision-language-navigation
git worktree remove .worktrees/batch150-document-vlm
git worktree remove .worktrees/batch150-image-to-video
git worktree remove .worktrees/batch150-image-to-3d
git worktree remove .worktrees/batch150-integration
```

Expected: only the main worktree remains.
