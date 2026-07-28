# 100 New Directions Batch Design

**Date:** 2026-04-09

## Goal

Add one new 100-family expansion batch built from **10 previously unimplemented directions**, with
**10 compact-first families per direction**, while avoiding new lesson work and avoiding new test files.

This batch is intentionally scoped to:

- new `dlhub/*` direction packages only
- local zoo / registry integration
- README count + direction table updates
- minimal verification only

This batch explicitly excludes:

- `tracks/*` lesson creation
- new dependencies
- full test authoring for the new directions

## Approved Direction Set

The following directions were selected because they are present in the user's topic pool and do not
already exist as first-class packages in the current repository:

1. `dlhub/vision/video_frame_interpolation/`
2. `dlhub/vision/video_stabilization/`
3. `dlhub/vision/video_object_detection/`
4. `dlhub/vision/document_dewarping/`
5. `dlhub/vision/layout_generation/`
6. `dlhub/vision/adversarial_robustness/`
7. `dlhub/vision/data_augmentation/`
8. `dlhub/vision/image_synthesis/`
9. `dlhub/multimodal/prompt_learning/`
10. `dlhub/pointcloud/gaussian_splatting/`

## Architecture

The batch will reuse the repository's existing compact-first family pattern:

- one direction package per domain
- one family per file
- `_VARIANTS` containing `tiny/small/base`
- one lazy-import package `__init__.py`
- one registry-backed zoo entry surface where the domain already uses that pattern

The implementation should follow the actual repository layout instead of inventing a new abstraction:

- `vision` directions get their own `dlhub/vision/<direction>_zoo.py` files
- `pointcloud` directions get a direction-specific zoo when that domain already follows a dedicated zoo pattern
- `multimodal` directions follow the lazy `dlhub/multimodal/*` package convention

## Naming Strategy

Each direction gets 10 families with names that match the repo's style:

### 1. Video Frame Interpolation

- `sepconv_interp`
- `super_slomo`
- `dain_baseline`
- `rife_baseline`
- `flavr_baseline`
- `vfi_former`
- `amt_interp`
- `ifrnet_baseline`
- `ema_vfi`
- `mamba_vfi`

### 2. Video Stabilization

- `deshake_net`
- `stabilizer_cnn`
- `deep_stab`
- `steady_flow`
- `warp_stab`
- `traj_stab`
- `gyro_stab`
- `mesh_stab`
- `transformer_stab`
- `mamba_stab`

### 3. Video Object Detection

- `fgfa_det`
- `selsa_det`
- `megan_det`
- `dff_det`
- `tubelet_det`
- `seqformer_det`
- `vitvod_det`
- `flowrcnn_vid`
- `trackdet_head`
- `mamba_vid_det`

### 4. Document Dewarping

- `docunet_warp`
- `dewarp_net`
- `scanner_rectify`
- `page_curve_net`
- `book_flatten`
- `mesh_dewarp`
- `textline_dewarp`
- `quad_rectifier`
- `docformer_dewarp`
- `mamba_dewarp`

### 5. Layout Generation

- `layoutgan_baseline`
- `layoutvae_baseline`
- `layouttransformer`
- `bbox_generator`
- `poster_layout_net`
- `doc_layout_gen`
- `constraint_layout`
- `relation_layout`
- `diffusion_layout`
- `mamba_layout_gen`

### 6. Adversarial Robustness

- `fgsm_guard`
- `pgd_guard`
- `trades_guard`
- `mart_guard`
- `free_at_guard`
- `fast_at_guard`
- `feature_denoise_guard`
- `adv_prop_guard`
- `patch_guard`
- `certified_guard`

### 7. Data Augmentation

- `mixup_aug`
- `cutmix_aug`
- `fmix_aug`
- `gridmask_aug`
- `randaugment_aug`
- `trivialaugment_aug`
- `autoaugment_aug`
- `augmix_aug`
- `mosaic_aug`
- `copy_paste_aug`

### 8. Image Synthesis

- `pix2pix_synth`
- `gaugan_synth`
- `cascaded_synth`
- `palette_synth`
- `control_synth`
- `latent_synth`
- `diffusion_synth`
- `prompt2img_synth`
- `layout2img_synth`
- `mamba_synth`

### 9. Prompt Learning

- `coop_prompt`
- `cocoop_prompt`
- `proda_prompt`
- `vpt_prompt`
- `promptsrc_prompt`
- `maple_prompt`
- `dapt_prompt`
- `adapter_prompt`
- `prefix_fusion_prompt`
- `mamba_promptlearn`

### 10. Gaussian Splatting

- `gaussian_splat`
- `mip_splat`
- `dynamic_splat`
- `surf_splat`
- `gs_anchor`
- `compact_splat`
- `deform_splat`
- `street_splat`
- `sparse_splat`
- `mamba_splat`

## Write Boundaries

Each direction owns only its own package plus the minimum central registry files needed for exposure.

Direction-local files:

- package `__init__.py`
- 10 family files
- optional package `README.md`
- direction-specific zoo file

Shared integration files:

- `README.md`
- one new direction-specific zoo file per new direction
- domain-level package exports where needed
- optional domain-level zoo scripts only if required for discoverability

## Verification Strategy

The user explicitly allowed skipping new tests, so this batch will use only minimal verification:

- import verification for each new package
- registry/list output spot checks
- builder smoke for one representative family per direction
- `git diff --check`

No new pytest files are required for this batch.

## Parallel Execution Model

This batch is designed for:

- 10 worktrees
- 10 implementation lanes
- 10 verification / registry lanes
- one final integration merge after all 100 families land

Each worktree owns exactly one direction to minimize merge conflicts.

## Merge Strategy

This batch should merge only after all 10 directions are complete and minimal verification passes.

The intended cadence is:

1. create 10 worktrees
2. implement 10 directions in parallel
3. verify each direction minimally
4. merge all 10 direction branches into one batch integration branch
5. merge the batch into `main`
6. repeat for the next unseen 100-family batch
