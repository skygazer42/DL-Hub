# Cross-Domain New Directions Batch 16 Design

**Date:** 2026-04-14

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
2. The batch should stay cross-domain instead of collapsing into a single-domain expansion.
3. Every direction must fit the existing `10 families + toy-first + tiny/small/base` contract.
4. Shared integration changes must stay on a final integration branch rather than leaking into
   direction branches.

## Approved Direction Set

The following directions were selected because they fit the user's approved cross-domain strategy
and do not already exist as first-class packages in the current repository:

1. `dlhub/vision/image_deweathering/`
2. `dlhub/vision/transparent_depth_estimation/`
3. `dlhub/pointcloud/pointcloud_forecasting/`
4. `dlhub/pointcloud/pointcloud_anomaly_detection/`
5. `dlhub/multimodal/video_text_retrieval/`
6. `dlhub/multimodal/embodied_question_answering/`
7. `dlhub/multimodal/audio_text_understanding/`
8. `dlhub/generative/text_to_video/`
9. `dlhub/generative/video_to_video/`
10. `dlhub/generative/world_models/`

This keeps the batch balanced while still shifting more capacity into the thinner domains:

- `vision`: 2 directions
- `pointcloud`: 2 directions
- `multimodal`: 3 directions
- `generative`: 3 directions

## Batch Strategy

This batch uses a **cross-domain balanced** strategy rather than a `vision`-heavy or purely
generative batch.

Reasons:

- it preserves the unseen-direction expansion contract already established by Batches 14 and 15
- it gives more of the batch budget to `multimodal` and `generative`, where the repo still has
  clearer first-class package gaps
- it still keeps each implementation lane mostly local to one directory tree
- it preserves the same README storytelling shape as the previous cross-domain batches

## Family Naming Strategy

Each direction gets 10 families that mix recognizable baseline names with repository-friendly
modern variants.

### 1. Image Deweathering

- `snow_removal`
- `raindrop_removal`
- `fog_streak_removal`
- `all_weather_restore`
- `deweather_cnn`
- `transformer_deweather`
- `frequency_deweather`
- `diffusion_deweather`
- `prompt_deweather`
- `mamba_deweather`

### 2. Transparent Depth Estimation

- `glass_depth`
- `refract_depth`
- `trimap_depth`
- `boundary_depth`
- `layered_depth`
- `transformer_transdepth`
- `geometry_transdepth`
- `diffusion_transdepth`
- `prompt_transdepth`
- `mamba_transdepth`

### 3. Point Cloud Forecasting

- `pointlstm_forecast`
- `trajpoint_forecast`
- `motion_field_forecast`
- `scene_memory_forecast`
- `graph_forecast3d`
- `transformer_forecast3d`
- `diffusion_forecast3d`
- `prompt_forecast3d`
- `occupancy_forecast3d`
- `mamba_forecast3d`

### 4. Point Cloud Anomaly Detection

- `recon_anomaly3d`
- `patchcore_anomaly3d`
- `student_teacher_anomaly3d`
- `memory_anomaly3d`
- `density_anomaly3d`
- `transformer_anomaly3d`
- `diffusion_anomaly3d`
- `prompt_anomaly3d`
- `openvocab_anomaly3d`
- `mamba_anomaly3d`

### 5. Video-Text Retrieval

- `clip4clip_retrieval`
- `xpool_retrieval`
- `frozen_retrieval`
- `dual_encoder_vtr`
- `cross_encoder_vtr`
- `temporal_align_vtr`
- `transformer_vtr`
- `retrieval_aug_vtr`
- `prompt_vtr`
- `mamba_vtr`

### 6. Embodied Question Answering

- `navqa_embodied`
- `memory_eqa`
- `objectnav_eqa`
- `mapqa_embodied`
- `speaker_eqa`
- `transformer_eqa`
- `grounded_eqa`
- `retrieval_eqa`
- `prompt_eqa`
- `mamba_eqa`

### 7. Audio-Text Understanding

- `audio_bert_understanding`
- `wav2text_understanding`
- `contrastive_atu`
- `event_audio_text`
- `speech_audio_text`
- `transformer_atu`
- `retrieval_atu`
- `diffusion_atu`
- `prompt_atu`
- `mamba_atu`

### 8. Text-to-Video

- `zeroscope_t2v`
- `modelscope_t2v`
- `lavie_t2v`
- `motion_prior_t2v`
- `cascade_t2v`
- `control_t2v`
- `transformer_t2v`
- `diffusion_t2v`
- `prompt_t2v`
- `mamba_t2v`

### 9. Video-to-Video

- `vid2vid_translation`
- `style_vid2vid`
- `tokenflow_vid2vid`
- `control_vid2vid`
- `temporal_consistency_v2v`
- `motion_transfer_v2v`
- `transformer_v2v`
- `diffusion_v2v`
- `prompt_v2v`
- `mamba_v2v`

### 10. World Models

- `rssm_world`
- `dreamer_world`
- `video_world`
- `latent_dynamics_world`
- `action_conditioned_world`
- `memory_world`
- `transformer_world`
- `diffusion_world`
- `prompt_world`
- `mamba_world`

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

- `image_deweathering`: `build_<family>_deweatherer(...)`
- `transparent_depth_estimation`: `build_<family>_transparent_depth_model(...)`
- `pointcloud_forecasting`: `build_<family>_forecasting_model(...)`
- `pointcloud_anomaly_detection`: `build_<family>_anomaly_detector(...)`
- `video_text_retrieval`: `build_<family>_retriever(...)`
- `embodied_question_answering`: `build_<family>_embodied_qa_model(...)`
- `audio_text_understanding`: `build_<family>_audio_text_model(...)`
- `text_to_video`: `build_<family>_text_to_video(...)`
- `video_to_video`: `build_<family>_video_to_video(...)`
- `world_models`: `build_<family>_world_model(...)`

## Zoo Strategy

Zoo style stays domain-specific rather than being normalized across all domains:

- `vision`: explicit `_FAMILIES + _SIZES` registry, with prefixes `deweather:` and `transdepth:`
- `multimodal`: explicit `_FAMILIES + _SIZES` registry, with prefixes `vtr:`, `eqa:`, and `atu:`
- `pointcloud`: AST-based discovery zoo, with prefixes `pcforecast:` and `pcanomaly:`
- `generative`: AST-based discovery zoo, with prefixes `t2v:`, `v2v:`, and `world:`

This matches the current repository patterns:

- explicit registries for `vision` and `multimodal`
- AST discovery for `pointcloud` and `generative`

## Worktree and Ownership Model

This batch is designed for one worktree per direction:

- `batch160-image-deweathering`
- `batch160-transparent-depth-estimation`
- `batch160-pointcloud-forecasting`
- `batch160-pointcloud-anomaly-detection`
- `batch160-video-text-retrieval`
- `batch160-embodied-question-answering`
- `batch160-audio-text-understanding`
- `batch160-text-to-video`
- `batch160-video-to-video`
- `batch160-world-models`

Then one integration branch:

- `batch160-integration`

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
- `git status --short --branch` on the integration branch before merge-back

No new pytest files are required for this batch.

## README Recording Strategy

`README.md` should receive exactly one additional "Additional New Directions" table:

- title: `Additional New Directions / 新增研究方向（十六）`
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
7. confirm the integration branch is clean with `git status --short --branch`
8. merge the integration branch back to `main`
9. remove the worktrees after merge
