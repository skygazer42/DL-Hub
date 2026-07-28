# Deraining Regression Tests Implementation Plan

**Goal:** Exercise the missing deraining model families in unit tests so their absence surfaces during regression.

**Architecture:** Add a regression-only smoke test that builds each deraining-family model from the synthetic denoising lesson, then assert backward compatibility with the rain-noise dataloader. Extend the listing/arch-family discovery tests so the CLI mentions the new families without touching production wiring.

**Tech Stack:** Python 3, `pytest`, `tracks.vision.lesson_10_synthetic_denoising`, CLI parsing utilities under `tracks.vision.lesson_10_synthetic_denoising.cli`.

---

### Task 1: Deraining regression tests only

**Files:**
- Modify: `F:/DL-Hub/.worktrees/vision-deraining-expansion/tests/test_tracks_vision_denoising.py`

**Step 1: Write the failing test**

```python
@pytest.mark.parametrize("family:model", [...])
def test_deraining_rain_models_forward_backward_smoke(...):
    config = DataConfig(..., noise_type="rain")
    dataloaders = get_dataloaders(config)
    model = build_model(model)
    result = model(batch)
    loss = torch.nn.functional.mse_loss(result, target)
    loss.backward()
```

Expected: test fails today because the enumerated families are not exposed in `build_model`.

**Step 2: Run the focused tests to verify failure**

Run: `python -m pytest tests/test_tracks_vision_denoising.py -q -k "deraining_rain_models_forward_backward_smoke or list_arch_families or arch_filters_by_family"`
Expected: failure citing missing `--arch-family` entries.

**Step 3: Describe future production work**

Outline that the same families need to be wired into the CLI helpers and `build_model`, but that work belongs to the follow-up task once the tests prove the gap.

**Step 4: Re-run the focused command to ensure the failure is still intentional**

Same command as Step 2 with the expectation still failing.

**Step 5: Commit**

Run:
```
git add tests/test_tracks_vision_denoising.py
git add docs/plans/2026-04-03-deraining-regression-tests.md
git commit -m "test: add deraining regression smoke coverage"
```
