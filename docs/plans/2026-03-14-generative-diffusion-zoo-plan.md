# Diffusion Zoo Implementation Plan

**Goal:** Add a local diffusion zoo with 12 families / 36 arches, timeline metadata, recommendation profiles, and a CLI smoke script.

**Architecture:** Mirror the existing `gan_zoo` pattern so each diffusion family lives in its own module with `_VARIANTS`, `build_<family>_diffusion(...)`, and a `__main__` smoke guard. Use a shared compact diffusion backbone in `dlhub/generative/diffusion/_common.py`, then layer registry, timeline, recommendation, and script utilities on top.

**Tech Stack:** Python, PyTorch, pytest, ruff

---

### Task 1: Lock the red test baseline

**Files:**
- Test: `tests/test_dlhub_generative_diffusion_zoo.py`
- Test: `tests/test_dlhub_generative_diffusion_timeline.py`
- Test: `tests/test_dlhub_generative_diffusion_algorithms.py`
- Test: `tests/test_dlhub_generative_diffusion_recommend.py`

**Step 1: Write the failing test**

The diffusion tests already exist and define the expected API surface.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_generative_diffusion_zoo.py tests/test_dlhub_generative_diffusion_timeline.py tests/test_dlhub_generative_diffusion_algorithms.py tests/test_dlhub_generative_diffusion_recommend.py`
Expected: FAIL with missing `dlhub.generative.diffusion_zoo` / `dlhub.generative.diffusion`.

**Step 3: Write minimal implementation**

Create the package skeleton under `dlhub/generative/diffusion/`, the zoo registry, and the CLI entrypoint.

**Step 4: Run test to verify it passes**

Run the same `pytest` command and expect the import errors to disappear.

### Task 2: Build the shared diffusion core and 12 family modules

**Files:**
- Create: `dlhub/generative/diffusion/_common.py`
- Create: `dlhub/generative/diffusion/ddpm.py`
- Create: `dlhub/generative/diffusion/ddim.py`
- Create: `dlhub/generative/diffusion/iddpm.py`
- Create: `dlhub/generative/diffusion/score_sde.py`
- Create: `dlhub/generative/diffusion/ncsnpp.py`
- Create: `dlhub/generative/diffusion/edm.py`
- Create: `dlhub/generative/diffusion/latent_diffusion.py`
- Create: `dlhub/generative/diffusion/stable_diffusion.py`
- Create: `dlhub/generative/diffusion/consistency_model.py`
- Create: `dlhub/generative/diffusion/flow_matching.py`
- Create: `dlhub/generative/diffusion/rectified_flow.py`
- Create: `dlhub/generative/diffusion/conditional_flow_matching.py`
- Test: `tests/test_zoo_conventions_smoke.py`

**Step 1: Write the failing test**

Use the existing algorithm smoke tests plus the convention smoke test as the contract.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_generative_diffusion_algorithms.py tests/test_zoo_conventions_smoke.py`
Expected: FAIL because builders and family files do not exist yet.

**Step 3: Write minimal implementation**

Implement one shared `CompactDiffusion` model and keep each family module thin.

**Step 4: Run test to verify it passes**

Run the same `pytest` command and expect all family builders and module conventions to pass.

### Task 3: Add metadata, recommendations, registry, and CLI

**Files:**
- Create: `dlhub/generative/diffusion/_timeline.py`
- Create: `dlhub/generative/diffusion/_recommend.py`
- Create: `dlhub/generative/diffusion/README.md`
- Create: `dlhub/generative/diffusion_zoo.py`
- Create: `scripts/diffusion_zoo.py`

**Step 1: Write the failing test**

Use the timeline and recommendation tests as the contract for groups, years, profiles, and CLI text.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_generative_diffusion_timeline.py tests/test_dlhub_generative_diffusion_recommend.py`
Expected: FAIL because timeline metadata and script outputs are missing.

**Step 3: Write minimal implementation**

Add four groups, recommendation profiles, `diff:` registry generation, and CLI commands `--list`, `--timeline`, `--list-profiles`, `--recommend`, and `--smoke`.

**Step 4: Run test to verify it passes**

Run the same `pytest` command and expect all metadata and CLI checks to pass.

### Task 4: Verify the completed feature

**Files:**
- Modify: `tests/test_zoo_conventions_smoke.py`
- Test: `tests/test_dlhub_generative_diffusion_zoo.py`
- Test: `tests/test_dlhub_generative_diffusion_timeline.py`
- Test: `tests/test_dlhub_generative_diffusion_algorithms.py`
- Test: `tests/test_dlhub_generative_diffusion_recommend.py`

**Step 1: Run lint**

Run: `ruff check dlhub/generative dlhub/generative/diffusion_zoo.py scripts/diffusion_zoo.py tests/test_dlhub_generative_diffusion_zoo.py tests/test_dlhub_generative_diffusion_timeline.py tests/test_dlhub_generative_diffusion_algorithms.py tests/test_dlhub_generative_diffusion_recommend.py tests/test_zoo_conventions_smoke.py`
Expected: PASS.

**Step 2: Run targeted tests**

Run: `pytest -q tests/test_dlhub_generative_diffusion_zoo.py tests/test_dlhub_generative_diffusion_timeline.py tests/test_dlhub_generative_diffusion_algorithms.py tests/test_dlhub_generative_diffusion_recommend.py tests/test_zoo_conventions_smoke.py`
Expected: PASS.

**Step 3: Run manual CLI smoke**

Run:
- `python scripts/diffusion_zoo.py --list --limit 5`
- `python scripts/diffusion_zoo.py --timeline`
- `python scripts/diffusion_zoo.py --recommend fidelity --variant tiny --top-k 4`

Expected: human-readable output with `Diffusion` headers and `diff:` arch ids.
