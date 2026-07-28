# Vision Denoising (DDPM U-Net) Implementation Plan

**Goal:** Add a **diffusion-style DDPM U-Net denoiser** (compact-first, pure PyTorch) under `dlhub/vision/denoising/`, wire it into the Lesson 10 synthetic denoising track, add tests, and push to `main`.

**Architecture:** Keep **one algorithm family per file** with variants declared via `_VARIANTS` and a `build_*_denoiser(...)` factory. The DDPM U-Net predicts a residual/noise estimate conditioned on a scalar `sigma`; the denoiser wrapper subtracts that residual from the noisy input (same conventions as DnCNN).

**Tech Stack:** Python, PyTorch, pytest, existing `tracks.vision.lesson_10_synthetic_denoising` training harness.

---

## Task 1: Add Track Smoke Test (TDD)

**Files:**
- Modify: `tests/test_tracks_vision_denoising.py`

**Step 1: Write failing test**
- Add `ddpm_unet:ddpm_unet_tiny` to the model loop in `test_vision_denoising_supervised_forward_loss_backward_smoke`.

**Step 2: Run test to verify RED**
- Run: `pytest -q tests/test_tracks_vision_denoising.py::test_vision_denoising_supervised_forward_loss_backward_smoke`
- Expected: FAIL with `ValueError: Unknown arch` because `build_model()` does not yet support `ddpm_unet`.

## Task 2: Implement DDPM U-Net Denoiser Family

**Files:**
- Create: `dlhub/vision/denoising/ddpm_unet.py`
- Modify: `dlhub/vision/denoising/__init__.py`

**Step 1: Add `DDPMUNet` backbone**
- Small U-Net with residual blocks and sinusoidal sigma embedding.

**Step 2: Add `DDPMUNetDenoiser` wrapper**
- `forward(noisy) -> denoised`, using `denoised = noisy - backbone(noisy, sigma=...)`.

**Step 3: Add `_VARIANTS` and `build_ddpm_unet_denoiser(...)`**
- Provide at least `ddpm_unet_tiny/small/base`.

**Step 4: Add `__main__` smoke**
- Random forward + backward on CPU.

## Task 3: Wire Into Lesson 10 Model Builder

**Files:**
- Modify: `tracks/vision/lesson_10_synthetic_denoising/model.py`
- Modify: `tracks/vision/lesson_10_synthetic_denoising/README.md`

**Steps:**
- Add `ddpm_unet` to `ModelConfig.arch` comment.
- Extend `list_supported_arches()` with `ddpm_unet:_VARIANTS`.
- Add `build_model()` branch to call `build_ddpm_unet_denoiser(in_channels=..., sigma=..., variant=...)`.
- Update README model list with a short diffusion/DDPM mention and example CLI invocation.

## Task 4: Verify GREEN + Ship

**Step 1: Run tests**
- Run: `pytest -q tests/test_tracks_vision_denoising.py`
- Optionally: `pytest -q` (full suite)

**Step 2: Commit**
- `git add docs/plans/2026-03-05-vision-denoising-ddpm-unet-plan.md dlhub/vision/denoising/ddpm_unet.py dlhub/vision/denoising/__init__.py tracks/vision/lesson_10_synthetic_denoising/model.py tracks/vision/lesson_10_synthetic_denoising/README.md tests/test_tracks_vision_denoising.py`
- `git commit -m "feat(vision): add ddpm u-net denoiser"`

**Step 3: Push**
- `git push origin main`

