# Cross-Domain New Directions Batch Design

**Date:** 2026-04-11

## Goal

Add one new 100-family expansion batch built from **10 previously unimplemented directions** across
`vision`, `pointcloud`, `multimodal`, and `generative`, with **10 toy-first families per direction**.

This batch is intentionally scoped to:

- new `dlhub/*` direction packages only
- local zoo / registry integration where the domain already follows that pattern
- one additional README direction table entry for the new batch
- minimal verification only

This batch explicitly excludes:

- `tracks/*` lesson creation or modification
- new dependencies
- new pytest files
- broad refactors of existing zoo infrastructure

## Approved Direction Set

The following directions were selected because they fit the user's topic pool and do not already
exist as first-class packages in the current repository:

1. `dlhub/vision/image_relighting/`
2. `dlhub/vision/transparent_object_segmentation/`
3. `dlhub/vision/video_matting/`
4. `dlhub/vision/event_camera_understanding/`
5. `dlhub/pointcloud/scene_flow/`
6. `dlhub/pointcloud/pointcloud_completion/`
7. `dlhub/multimodal/audio_visual_learning/`
8. `dlhub/multimodal/multimodal_reasoning/`
9. `dlhub/generative/video_diffusion/`
10. `dlhub/generative/text_to_3d/`

## Batch Strategy

This batch uses a **cross-domain hybrid** strategy rather than a pure `vision` batch.

Reasons:

- it preserves the user's preference for unseen directions
- it gives clean landing zones for `3D`, `multimodal`, `prompt`, `AIGC`, and `diffusion`
- it keeps each implementation lane mostly local to one directory tree

The batch should still preserve the existing repository conventions:

- one direction package per domain
- one family per file
- `_VARIANTS` containing `tiny/small/base`
- one lazy-import package `__init__.py`
- small domain-local registry or zoo files only where the domain already benefits from them

## Naming Strategy

Each direction gets 10 families. The names should stay close to existing repository naming
conventions and deliberately mix classic baselines with modern variants.

### 1. Image Relighting

- `deep_relight`
- `hdr_relight`
- `intrinsic_relight`
- `ratio_relight`
- `retinex_relight`
- `portrait_relight`
- `transformer_relight`
- `diffusion_relight`
- `prompt_relight`
- `mamba_relight`

### 2. Transparent Object Segmentation

- `glassseg_toy`
- `translab_seg`
- `refractmask_seg`
- `camotransparent_seg`
- `trimap_transparent`
- `boundary_glass_seg`
- `transformer_transparent`
- `diffusion_transparent`
- `prompt_transparent`
- `mamba_transparent`

### 3. Video Matting

- `dim_vmatte`
- `fba_vmatte`
- `rvm_toy`
- `gca_vmatte`
- `tcvomatting`
- `memory_vmatte`
- `transformer_vmatte`
- `flow_vmatte`
- `prompt_vmatte`
- `mamba_vmatte`

### 4. Event Camera Understanding

- `ev_cnn`
- `voxel_eventnet`
- `spike_eventnet`
- `event_unet`
- `event_tracker`
- `event_depth`
- `transformer_event`
- `state_space_event`
- `crossmodal_event`
- `mamba_event`

### 5. Scene Flow

- `flow3d_pointnet`
- `flow3d_flownet`
- `pointpwc_flow`
- `raft3d_points`
- `iter_flow3d`
- `cost_volume_flow3d`
- `transformer_flow3d`
- `diffusion_flow3d`
- `prompt_flow3d`
- `mamba_flow3d`

### 6. Point Cloud Completion

- `pcn_completion`
- `topnet_completion`
- `grnet_completion`
- `snowflake_completion`
- `folding_completion`
- `anchor_completion`
- `transformer_completion`
- `diffusion_completion`
- `text_guided_completion`
- `mamba_completion`

### 7. Audio-Visual Learning

- `av_syncnet`
- `av_contrast`
- `av_fusionnet`
- `av_localizer`
- `av_separation`
- `av_caption_bridge`
- `transformer_av`
- `diffusion_av`
- `prompt_av`
- `mamba_av`

### 8. Multimodal Reasoning

- `vqa_reasoner`
- `chain_reasoner`
- `tool_reasoner`
- `memory_reasoner`
- `grounded_reasoner`
- `program_reasoner`
- `transformer_reasoner`
- `retrieval_reasoner`
- `prompt_reasoner`
- `mamba_reasoner`

### 9. Video Diffusion

- `latent_video_diffusion`
- `frame_interp_diffusion`
- `video_unet_diffusion`
- `cascade_video_diffusion`
- `motion_prior_diffusion`
- `control_video_diffusion`
- `transformer_video_diffusion`
- `rectified_video_diffusion`
- `prompt_video_diffusion`
- `mamba_video_diffusion`

### 10. Text-to-3D

- `dreamfusion_toy`
- `magic3d_toy`
- `score_distill_3d`
- `neural_lift_3d`
- `sdf_prompt_3d`
- `mesh_diffuse_3d`
- `transformer_text3d`
- `gaussian_text3d`
- `layout_text3d`
- `mamba_text3d`

## File Layout

Each new direction should create the minimum complete package:

- `__init__.py`
- `README.md`
- 10 family files

Additional domain-local zoo files are expected for the new non-vision directions where direction
discovery should remain explicit and low-coupling:

- `dlhub/pointcloud/scene_flow_zoo.py`
- `dlhub/pointcloud/pointcloud_completion_zoo.py`
- `dlhub/multimodal/audio_visual_learning_zoo.py`
- `dlhub/multimodal/multimodal_reasoning_zoo.py`
- `dlhub/generative/video_diffusion_zoo.py`
- `dlhub/generative/text_to_3d_zoo.py`

For `vision`, prefer reusing the existing local discovery surface rather than inventing a new
direction-specific zoo unless the current discovery path proves insufficient.

## Worktree and Ownership Model

This batch is designed for one worktree per direction:

- `batch140-image-relighting`
- `batch140-transparent-object-segmentation`
- `batch140-video-matting`
- `batch140-event-camera-understanding`
- `batch140-scene-flow`
- `batch140-pointcloud-completion`
- `batch140-audio-visual-learning`
- `batch140-multimodal-reasoning`
- `batch140-video-diffusion`
- `batch140-text-to-3d`

Then one integration branch:

- `batch140-integration`

Direction worktrees should only own:

- the direction package
- the direction README
- the direction-local zoo file if that zoo belongs to the same direction

The integration branch should own:

- `README.md`
- shared package exports
- final registry or zoo wiring that touches shared files

## Verification Strategy

The user explicitly allowed skipping new tests, so this batch will use only minimal verification:

- one import/build smoke per new direction
- one list / discovery check per new zoo file
- one final cross-domain smoke covering `vision`, `pointcloud`, `multimodal`, and `generative`
- `git diff --check`

No new pytest files are required for this batch.

## README Recording Strategy

`README.md` should receive one additional "Additional New Directions" table only.

Do not rewrite older batch sections unless required for count consistency. The intent is to keep
the historical progression append-only and make the next batch equally easy to record.

## Merge Strategy

This batch should merge only after all 10 directions are complete and minimal verification passes.

The intended cadence is:

1. create 10 worktrees
2. implement 10 directions in parallel
3. run direction-level minimal verification
4. merge the 10 direction branches into one integration branch
5. land integration changes and merge the batch into `main`
6. remove worktrees after merge
