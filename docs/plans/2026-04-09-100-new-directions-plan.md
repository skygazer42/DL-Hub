# 100 New Directions Batch Implementation Plan

**Goal:** Add 100 new compact-first algorithm families across 10 previously unimplemented directions without adding lessons or new test files.

**Architecture:** Create 8 new `vision` direction packages, 1 new `multimodal` direction package, and 1 new `pointcloud` direction package. Reuse the repository's lazy-import package structure and existing direction-specific zoo conventions so each new direction gets its own discovery/build surface with minimal shared-file contention.

**Tech Stack:** Python 3, `torch`, repo-local compact model helpers, direction-specific `*_zoo.py` files, `README.md`, lazy packages under `dlhub/vision`, `dlhub/multimodal`, and `dlhub/pointcloud`.

---

### Task 1: Add video frame interpolation direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/sepconv_interp.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/super_slomo.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/dain_baseline.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/rife_baseline.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/flavr_baseline.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/vfi_former.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/amt_interp.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/ifrnet_baseline.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/ema_vfi.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/mamba_vfi.py`
- Create: `F:/DL-Hub/dlhub/vision/video_frame_interpolation/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package lazy importer**

Follow the same lazy `__getattr__` pattern already used by recent direction packages under `dlhub/vision/`.

**Step 2: Add 10 family files**

Each file must:

- define `_VARIANTS` with `tiny/small/base`
- expose one `build_<family>_...` function
- follow compact-first output contracts appropriate for interpolation
- include a `__main__` smoke path

**Step 3: Wire the new families into `dlhub/vision/local_zoo.py`**

Add lazy registry discovery so the 10 variants show up through normal `vision` listing/build flows.

**Step 4: Update README direction table**

Add the new direction with family count `10`.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.video_frame_interpolation.sepconv_interp import build_sepconv_interp_interpolator as f; print(type(f(in_channels=3, variant='sepconv_interp_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('sepconv_interp_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and local zoo reports at least one new frame interpolation variant.

### Task 2: Add video stabilization direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/deshake_net.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/stabilizer_cnn.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/deep_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/steady_flow.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/warp_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/traj_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/gyro_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/mesh_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/transformer_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/mamba_stab.py`
- Create: `F:/DL-Hub/dlhub/vision/video_stabilization/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package lazy importer**

Reuse the existing `vision` lazy package pattern.

**Step 2: Add 10 family files**

Keep outputs compact-first and stabilization-oriented. Do not add extra shared abstractions unless at least 3 files need the same helper.

**Step 3: Wire into `dlhub/vision/local_zoo.py`**

Expose the new families via the main `vision` listing/build surface.

**Step 4: Update README**

Add the direction and count.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.video_stabilization.deshake_net import build_deshake_net_stabilizer as f; print(type(f(in_channels=3, variant='deshake_net_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('deshake_net_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and at least one stabilization variant is visible in the local zoo.

### Task 3: Add video object detection direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/fgfa_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/selsa_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/megan_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/dff_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/tubelet_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/seqformer_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/vitvod_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/flowrcnn_vid.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/trackdet_head.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/mamba_vid_det.py`
- Create: `F:/DL-Hub/dlhub/vision/video_object_detection/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Use the same lazy importer shape as other `vision` direction packages.

**Step 2: Add 10 detector family files**

Each family should return a stable compact detection-style output without introducing external datasets or heavy dependencies.

**Step 3: Wire into `vision` local zoo**

Keep names consistent with existing arch-id style.

**Step 4: Update README**

Add the new direction row.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.video_object_detection.fgfa_det import build_fgfa_det_video_detector as f; print(type(f(in_channels=3, num_classes=4, variant='fgfa_det_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('fgfa_det_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and zoo listing exposes at least one video object detection variant.

### Task 4: Add document dewarping direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/docunet_warp.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/dewarp_net.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/scanner_rectify.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/page_curve_net.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/book_flatten.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/mesh_dewarp.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/textline_dewarp.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/quad_rectifier.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/docformer_dewarp.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/mamba_dewarp.py`
- Create: `F:/DL-Hub/dlhub/vision/document_dewarping/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Add the lazy package entrypoint.

**Step 2: Add 10 dewarping family files**

Model outputs should remain compact-first and OCR/document friendly.

**Step 3: Integrate with `vision` local zoo**

Expose the new variants through normal listing/build flows.

**Step 4: Update README**

Add the new direction row.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.document_dewarping.docunet_warp import build_docunet_warp_dewarper as f; print(type(f(in_channels=3, variant='docunet_warp_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('docunet_warp_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and zoo listing exposes a document dewarping variant.

### Task 5: Add layout generation direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/layoutgan_baseline.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/layoutvae_baseline.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/layouttransformer.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/bbox_generator.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/poster_layout_net.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/doc_layout_gen.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/constraint_layout.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/relation_layout.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/diffusion_layout.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/mamba_layout_gen.py`
- Create: `F:/DL-Hub/dlhub/vision/layout_generation/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Use lazy import conventions.

**Step 2: Add 10 layout generation families**

Keep compact outputs stable and document / AIGC oriented.

**Step 3: Add registry wiring**

Expose them via the main `vision` local zoo.

**Step 4: Update README**

Add the direction row and count.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.layout_generation.layoutgan_baseline import build_layoutgan_baseline_layout_generator as f; print(type(f(in_channels=3, variant='layoutgan_baseline_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('layoutgan_baseline_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and zoo listing exposes at least one layout generation variant.

### Task 6: Add adversarial robustness direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/fgsm_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/pgd_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/trades_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/mart_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/free_at_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/fast_at_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/feature_denoise_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/adv_prop_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/patch_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/certified_guard.py`
- Create: `F:/DL-Hub/dlhub/vision/adversarial_robustness/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Use the normal lazy import style.

**Step 2: Add 10 robustness families**

These may wrap or adapt compact classifiers, but must not add external libraries or actual attack pipelines beyond compact-first abstractions.

**Step 3: Wire into `vision` local zoo**

Expose arch ids normally.

**Step 4: Update README**

Add the direction row.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.adversarial_robustness.fgsm_guard import build_fgsm_guard_robust_model as f; print(type(f(in_channels=3, num_classes=4, variant='fgsm_guard_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('fgsm_guard_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and zoo listing exposes a robustness variant.

### Task 7: Add data augmentation direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/mixup_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/cutmix_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/fmix_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/gridmask_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/randaugment_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/trivialaugment_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/autoaugment_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/augmix_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/mosaic_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/copy_paste_aug.py`
- Create: `F:/DL-Hub/dlhub/vision/data_augmentation/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Use the same lazy package approach.

**Step 2: Add 10 augmentation family files**

Treat each family as a compact-first transform model / augmentation policy builder consistent with repo conventions.

**Step 3: Wire the direction into `vision` local zoo**

Expose these through the same arch-id pathway.

**Step 4: Update README**

Add the direction row.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.data_augmentation.mixup_aug import build_mixup_aug_augmenter as f; print(type(f(in_channels=3, variant='mixup_aug_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('mixup_aug_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and zoo listing exposes an augmentation variant.

### Task 8: Add image synthesis direction

**Files:**
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/__init__.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/pix2pix_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/gaugan_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/cascaded_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/palette_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/control_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/latent_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/diffusion_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/prompt2img_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/layout2img_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/mamba_synth.py`
- Create: `F:/DL-Hub/dlhub/vision/image_synthesis/README.md`
- Modify: `F:/DL-Hub/dlhub/vision/local_zoo.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Use lazy imports.

**Step 2: Add 10 synthesis family files**

Keep them compact-first and AIGC-oriented without trying to reproduce heavy generation stacks.

**Step 3: Wire into `vision` local zoo**

Expose arch ids via the main local listing/build path.

**Step 4: Update README**

Add the direction row.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.vision.image_synthesis.pix2pix_synth import build_pix2pix_synth_generator as f; print(type(f(in_channels=3, variant='pix2pix_synth_tiny')).__name__)"
python -c "from dlhub.vision.local_zoo import list_local_arches; print(any('pix2pix_synth_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and zoo listing exposes a synthesis variant.

### Task 9: Add multimodal prompt learning direction

**Files:**
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/__init__.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/coop_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/cocoop_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/proda_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/vpt_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/promptsrc_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/maple_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/dapt_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/adapter_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/prefix_fusion_prompt.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/mamba_promptlearn.py`
- Create: `F:/DL-Hub/dlhub/multimodal/prompt_learning/README.md`
- Modify: `F:/DL-Hub/dlhub/multimodal/__init__.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Match the lazy `multimodal` package conventions already used by `vlm`.

**Step 2: Add 10 prompt learning family files**

Each file should expose one builder with compact-first multimodal prompt outputs.

**Step 3: Add multimodal package export wiring**

Expose the new package at the domain level without inventing a second registry abstraction unless required by actual repo patterns.

**Step 4: Update README**

Add the new direction row with count `10`.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.multimodal.prompt_learning.coop_prompt import build_coop_prompt_prompt_learner as f; print(type(f(in_channels=3, variant='coop_prompt_tiny')).__name__)"
python -c "import dlhub.multimodal.prompt_learning as p; print(hasattr(p, 'build_coop_prompt_prompt_learner'))"
```

Expected: import succeeds and the lazy multimodal package exposes at least one prompt learning builder.

### Task 10: Add pointcloud gaussian splatting direction

**Files:**
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/__init__.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/gaussian_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/mip_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/dynamic_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/surf_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/gs_anchor.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/compact_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/deform_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/street_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/sparse_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/mamba_splat.py`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting/README.md`
- Create: `F:/DL-Hub/dlhub/pointcloud/gaussian_splatting_zoo.py`
- Modify: `F:/DL-Hub/dlhub/pointcloud/__init__.py`
- Modify: `F:/DL-Hub/README.md`

**Step 1: Create the package**

Match the lazy package style already used by `dlhub/pointcloud/registration`.

**Step 2: Add 10 gaussian splatting family files**

Keep the files compact-first and stable. Do not attempt full renderer fidelity.

**Step 3: Create a small `gaussian_splatting_zoo.py`**

Follow the same pattern used by pointcloud domain-specific zoos such as `tracking3d_zoo.py`.

**Step 4: Update pointcloud package exports and README**

Expose the new direction and add the direction row.

**Step 5: Run minimal verification**

Run:

```bash
python -c "from dlhub.pointcloud.gaussian_splatting.gaussian_splat import build_gaussian_splat_splatter as f; print(type(f(in_channels=3, variant='gaussian_splat_tiny')).__name__)"
python -c "from dlhub.pointcloud.gaussian_splatting_zoo import list_local_arches; print(any('gaussian_splat_tiny' in x for x in list_local_arches()))"
```

Expected: import succeeds and the pointcloud gaussian splatting zoo lists at least one new variant.

### Task 11: Batch integration and verification

**Files:**
- Modify: `F:/DL-Hub/README.md`
- Modify: any shared registry or package export files touched by Tasks 1-10

**Step 1: Re-read the diff**

Confirm each worktree only touched its owned direction plus the minimal central integration files.

**Step 2: Run minimal cross-batch verification**

Run:

```bash
python -c "from dlhub.vision.local_zoo import list_local_arches; xs=list_local_arches(); print(sum(int(any(k in x for k in ['sepconv_interp_tiny','deshake_net_tiny','fgfa_det_tiny','docunet_warp_tiny','layoutgan_baseline_tiny','fgsm_guard_tiny','mixup_aug_tiny','pix2pix_synth_tiny'])) for x in xs))"
python -c "from dlhub.multimodal.prompt_learning.coop_prompt import build_coop_prompt_prompt_learner; print('ok')"
python -c "from dlhub.pointcloud.gaussian_splatting_zoo import list_local_arches; print(any('gaussian_splat_tiny' in x for x in list_local_arches()))"
git diff --check
```

Expected:

- new `vision` variants visible
- multimodal prompt learner imports
- pointcloud gaussian splatting zoo lists new variants
- `git diff --check` reports no patch formatting issues

**Step 3: Merge the 10 worktree branches into one batch branch**

Use non-interactive `git merge` only.

**Step 4: Merge the batch branch into `main`**

Use a non-interactive merge with a Lore-format commit message.

**Step 5: Record the completed batch in README**

Add one new section describing the 10 directions and the total 100-family expansion.
