# Cross-Domain New Directions Batch 15 Design

**Date:** 2026-04-13

## Goal

Add one new 100-family expansion batch built from **10 previously unimplemented directions** across
`vision`, `pointcloud`, `multimodal`, and `generative`, with **10 toy-first families per direction**.

This batch is intentionally scoped to:

- new `dlhub/*` direction packages only
- direction-local zoo or registry integration
- one additional README batch table entry
- minimal verification only

This batch explicitly excludes:

- `tracks/*` lesson creation or modification
- new dependencies
- new pytest files
- broad refactors of existing zoo infrastructure

## Direction Selection Rules

The batch must follow four rules:

1. Every chosen direction must be absent as a first-class package in the current repository.
2. The batch should stay cross-domain instead of collapsing into a `vision`-only expansion.
3. Every direction must fit the existing `10 families + toy-first + tiny/small/base` contract.
4. Shared integration changes must stay on a final integration branch rather than leaking into
   direction branches.

## Approved Direction Set

The following directions were selected because they fit the user's topic pool and do not already
exist as first-class packages in the current repository:

1. `dlhub/vision/image_deraining/`
2. `dlhub/vision/shadow_detection/`
3. `dlhub/pointcloud/pointcloud_upsampling/`
4. `dlhub/pointcloud/shape_correspondence_3d/`
5. `dlhub/pointcloud/open_vocabulary_3d/`
6. `dlhub/multimodal/image_text_retrieval/`
7. `dlhub/multimodal/vision_language_navigation/`
8. `dlhub/multimodal/document_vlm/`
9. `dlhub/generative/image_to_video/`
10. `dlhub/generative/image_to_3d/`

This keeps the batch balanced:

- `vision`: 2 directions
- `pointcloud`: 3 directions
- `multimodal`: 3 directions
- `generative`: 2 directions

## Batch Strategy

This batch uses a **cross-domain balanced** strategy rather than a pure `vision` batch or a purely
generative batch.

Reasons:

- it continues the user's preference for unseen directions
- it gives more of the batch budget to `3D`, `multimodal`, and `AIGC`
- it keeps each direction lane mostly local to one package tree
- it remains easy to integrate because the shared surface is still small

## Family Naming Strategy

Each direction gets 10 families that mix recognizable baseline names with repository-friendly
modern variants.

### 1. Image Deraining

- `jorder_derain`
- `did_mdn_derain`
- `resguide_derain`
- `recurrent_derain`
- `density_derain`
- `transformer_derain`
- `frequency_derain`
- `diffusion_derain`
- `prompt_derain`
- `mamba_derain`

### 2. Shadow Detection

- `dsd_shadow`
- `bdrar_shadow`
- `stacked_shadow`
- `context_shadow`
- `boundary_shadow`
- `transformer_shadow`
- `diffusion_shadow`
- `prompt_shadow`
- `state_space_shadow`
- `mamba_shadow`

### 3. Point Cloud Upsampling

- `punet_upsample`
- `mpu_upsample`
- `pugan_upsample`
- `dispu_upsample`
- `patch_upsample`
- `folding_upsample`
- `transformer_upsample`
- `diffusion_upsample`
- `prompt_upsample`
- `mamba_upsample`

### 4. Shape Correspondence 3D

- `fmnet_corr3d`
- `geodesic_corr3d`
- `descriptor_corr3d`
- `deformation_corr3d`
- `graphmatch_corr3d`
- `cycle_corr3d`
- `transformer_corr3d`
- `diffusion_corr3d`
- `prompt_corr3d`
- `mamba_corr3d`

### 5. Open Vocabulary 3D

- `clip3d_openvocab`
- `regionclip3d`
- `grounding3d_openvocab`
- `retrieval3d_openvocab`
- `proposal3d_openvocab`
- `languagefield_3d`
- `transformer_openvocab3d`
- `diffusion_openvocab3d`
- `prompt_openvocab3d`
- `mamba_openvocab3d`

### 6. Image-Text Retrieval

- `clip_retrieval`
- `albef_retrieval`
- `blip_retrieval`
- `dual_encoder_retrieval`
- `cross_attention_retrieval`
- `region_text_retrieval`
- `transformer_retrieval`
- `prompt_retrieval`
- `diffusion_retrieval`
- `mamba_retrieval`

### 7. Vision-Language Navigation

- `seq2seq_nav`
- `speaker_follower_nav`
- `map_memory_nav`
- `object_goal_nav`
- `panoramic_nav`
- `transformer_nav`
- `grounding_nav`
- `prompt_nav`
- `diffusion_nav`
- `mamba_nav`

### 8. Document VLM

- `layoutlm_doc`
- `docformer_doc`
- `pix2struct_doc`
- `donut_doc`
- `table_doc`
- `chart_doc`
- `transformer_doc`
- `retrieval_doc`
- `prompt_doc`
- `mamba_doc`

### 9. Image-to-Video

- `i2vgen_toy`
- `lavie_toy`
- `dynami_toy`
- `motion_adapter_i2v`
- `temporal_unet_i2v`
- `cascade_i2v`
- `control_i2v`
- `transformer_i2v`
- `diffusion_i2v`
- `mamba_i2v`

### 10. Image-to-3D

- `zero123_toy`
- `triplane_i23d`
- `lift3d_i23d`
- `mesh_i23d`
- `gaussian_i23d`
- `sdf_i23d`
- `transformer_i23d`
- `diffusion_i23d`
- `prompt_i23d`
- `mamba_i23d`

## File Layout and Interface Contract

Each new direction should create the minimum complete package:

- `__init__.py`
- `README.md`
- 10 family files

`_common.py` is allowed only when at least three files obviously share the same helper logic.

Each family file must follow the same contract:

- define `_VARIANTS` for `tiny`, `small`, and `base`
- expose exactly one public builder
- remain toy-first and CPU-friendly
- include a `__main__` smoke path

## Builder Suffixes

To keep lazy import logic predictable, each direction gets one fixed builder suffix:

- `image_deraining`: `build_<family>_derainer(...)`
- `shadow_detection`: `build_<family>_shadow_detector(...)`
- `pointcloud_upsampling`: `build_<family>_upsampler(...)`
- `shape_correspondence_3d`: `build_<family>_shape_correspondence_model(...)`
- `open_vocabulary_3d`: `build_<family>_open_vocabulary_3d_model(...)`
- `image_text_retrieval`: `build_<family>_retriever(...)`
- `vision_language_navigation`: `build_<family>_navigator(...)`
- `document_vlm`: `build_<family>_document_vlm(...)`
- `image_to_video`: `build_<family>_image_to_video(...)`
- `image_to_3d`: `build_<family>_image_to_3d_generator(...)`

## Zoo Strategy

Zoo style stays domain-specific rather than being normalized across all domains:

- `vision`: explicit `_FAMILIES + _SIZES` registry, with prefixes `derain:` and `shadow:`
- `multimodal`: explicit `_FAMILIES + _SIZES` registry, with prefixes `itr:`, `vln:`, and
  `docvlm:`
- `pointcloud`: AST-based discovery zoo, with prefixes `pcup:`, `s3corr:`, and `ov3d:`
- `generative`: AST-based discovery zoo, with prefixes `i2v:` and `i23d:`

This matches the current repository patterns:

- explicit registries for `vision` and `multimodal`
- AST discovery for `pointcloud` and `generative`

## Worktree and Ownership Model

This batch is designed for one worktree per direction:

- `batch150-image-deraining`
- `batch150-shadow-detection`
- `batch150-pointcloud-upsampling`
- `batch150-shape-correspondence-3d`
- `batch150-open-vocabulary-3d`
- `batch150-image-text-retrieval`
- `batch150-vision-language-navigation`
- `batch150-document-vlm`
- `batch150-image-to-video`
- `batch150-image-to-3d`

Then one integration branch:

- `batch150-integration`

Direction worktrees should only own:

- the direction package
- the direction README
- the direction-local zoo file

The integration branch should own:

- `README.md`
- `dlhub/pointcloud/__init__.py`
- `dlhub/multimodal/__init__.py`
- `dlhub/generative/__init__.py`

`vision` direction zoo files should remain local surfaces. They do not need a new shared export if
current repository conventions do not already expose them centrally.

## Verification Strategy

The user explicitly allowed skipping new tests, so this batch will use only minimal verification:

- one import/build smoke per new direction
- one list/discovery check per new zoo file
- one final cross-domain smoke covering `vision`, `pointcloud`, `multimodal`, and `generative`
- `git diff --check`

No new pytest files are required for this batch.

## README Recording Strategy

`README.md` should receive exactly one additional "Additional New Directions" table:

- title: `Additional New Directions / 新增研究方向（十五）`
- 10 rows, one for each approved direction
- append-only behavior, without rewriting older batch sections

## Merge Strategy

This batch should merge only after all 10 directions are complete and minimal verification passes.

The intended cadence is:

1. create 10 direction worktrees plus the integration worktree
2. implement the 10 directions in parallel
3. run direction-level smoke checks
4. merge the 10 direction branches into the integration branch
5. land shared exports and the README batch table
6. run final cross-domain smoke plus `git diff --check`
7. merge the integration branch back to `main`
8. remove the worktrees after merge
