# Vision Instance Segmentation 40-Family Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the local vision instance segmentation module to 40 distinct algorithm families with a discoverable local zoo, smoke coverage, and pure-torch toy-first builders.

**Architecture:** Keep the existing one-file-per-family convention under `dlhub/vision/instance_segmentation/`, add only the missing 19 families needed to reach 40 total, and introduce a lazy `instance_segmentation_zoo` that discovers `_VARIANTS` and `build_*_instance_segmenter(...)` factories from source. Reuse shared toy-first blocks in `_common.py` so the new families stay lightweight, distinct, and CPU-friendly while matching the repo's existing output patterns.

**Tech Stack:** Python, PyTorch, pytest, AST-based local zoo discovery, existing `dlhub.vision.instance_segmentation` module conventions.

---

### Task 1: Add failing zoo-level coverage for 40 instance segmentation families

**Files:**
- Create: `tests/test_dlhub_vision_instance_segmentation_zoo.py`
- Modify: `tests/test_dlhub_vision_instance_segmentation_algorithms.py`
- Modify: `tests/test_zoo_conventions_smoke.py`

**Step 1: Write the failing test**

Add a new zoo test that:
- imports `dlhub.vision.instance_segmentation_zoo`
- asserts `list_local_arches()` returns at least `120` ids
- asserts representative new ids exist:
  - `dlinst:deepmask_tiny`
  - `dlinst:sharpmask_tiny`
  - `dlinst:mnc_tiny`
  - `dlinst:instancefcn_tiny`
  - `dlinst:sipmask_tiny`
  - `dlinst:mask_dino_tiny`
  - `dlinst:deepsnake_tiny`
- parametrizes over all `dlinst:*_tiny` ids and runs forward/backward smoke through the zoo builder.

Update the algorithm smoke test so it also exercises representative builders from the new groups:
- `build_deepmask_instance_segmenter`
- `build_sipmask_instance_segmenter`
- `build_mask_dino_instance_segmenter`
- `build_deepsnake_instance_segmenter`

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_zoo.py tests/test_dlhub_vision_instance_segmentation_algorithms.py`

Expected:
- import failure for missing `dlhub.vision.instance_segmentation_zoo`, or
- missing builders / missing arch ids / too few arches

**Step 3: Write minimal implementation hooks**

Add the discovery layer and placeholder exports only as needed to move the tests from import errors to concrete missing-family failures.

**Step 4: Run test to verify it still fails for the right reason**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_zoo.py tests/test_dlhub_vision_instance_segmentation_algorithms.py`

Expected:
- failures now point at missing family modules or missing exports, not missing test infrastructure

**Step 5: Commit**

```bash
git add tests/test_dlhub_vision_instance_segmentation_zoo.py tests/test_dlhub_vision_instance_segmentation_algorithms.py tests/test_zoo_conventions_smoke.py
git commit -m "test: add instance segmentation zoo coverage"
```

### Task 2: Add the instance segmentation zoo and CLI wiring

**Files:**
- Create: `dlhub/vision/instance_segmentation_zoo.py`
- Create: `scripts/instance_segmentation_zoo.py`

**Step 1: Write the failing test**

Use Task 1's zoo test as the red test. Do not write production code until the new zoo import and API are failing in a targeted way.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_zoo.py::test_instance_segmentation_zoo_list_and_build_smoke`

Expected:
- `ModuleNotFoundError` or missing `list_local_arches/build_local_model`

**Step 3: Write minimal implementation**

Mirror the `detection_zoo.py` pattern:
- AST-extract `_VARIANTS`
- AST-extract the first `build_*_instance_segmenter` function
- lazy import builder on first use
- support `dlinst:` ids
- expose `BuildConfig`, `UnknownLocalArch`, `list_local_arches`, `build_local_model`

Add CLI support in `scripts/instance_segmentation_zoo.py`:
- `--list`
- `--search`
- `--smoke`

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_zoo.py::test_instance_segmentation_zoo_list_and_build_smoke`

Expected:
- import works
- test still fails because the total family count is below 40

**Step 5: Commit**

```bash
git add dlhub/vision/instance_segmentation_zoo.py scripts/instance_segmentation_zoo.py
git commit -m "feat: add instance segmentation local zoo"
```

### Task 3: Extend shared toy-first instance segmentation primitives

**Files:**
- Modify: `dlhub/vision/instance_segmentation/_common.py`

**Step 1: Write the failing test**

Use the representative builder smoke cases from Task 1 as the red tests for missing shared primitives required by new families.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "deepmask or sipmask or mask_dino or deepsnake"`

Expected:
- import failures or missing modules / missing layers needed by those builders

**Step 3: Write minimal implementation**

Add reusable helpers only where repetition would otherwise be obvious:
- proposal/query/contour heads
- pyramid backbone blocks
- basis/prototype helpers
- upsample / shape validation utilities

Keep the helpers small and generic enough for multiple families.

**Step 4: Run test to verify it passes or fails later in builder-specific code**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "deepmask or sipmask or mask_dino or deepsnake"`

Expected:
- helper-layer errors gone
- remaining failures are family-specific

**Step 5: Commit**

```bash
git add dlhub/vision/instance_segmentation/_common.py
git commit -m "refactor: extend instance segmentation shared blocks"
```

### Task 4: Implement classic and proposal-based families

**Files:**
- Create: `dlhub/vision/instance_segmentation/deepmask.py`
- Create: `dlhub/vision/instance_segmentation/sharpmask.py`
- Create: `dlhub/vision/instance_segmentation/cfm.py`
- Create: `dlhub/vision/instance_segmentation/mnc.py`
- Create: `dlhub/vision/instance_segmentation/instancefcn.py`
- Create: `dlhub/vision/instance_segmentation/panet.py`
- Create: `dlhub/vision/instance_segmentation/shapemask.py`
- Create: `dlhub/vision/instance_segmentation/bcnet.py`
- Create: `dlhub/vision/instance_segmentation/refinemask.py`
- Modify: `dlhub/vision/instance_segmentation/__init__.py`

**Step 1: Write the failing test**

Add the new builder entries to the algorithm smoke test and assert their zoo ids appear in the zoo test.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "deepmask or sharpmask or cfm or mnc or instancefcn or panet or shapemask or bcnet or refinemask"`

Expected:
- missing imports / unknown builders

**Step 3: Write minimal implementation**

For each family:
- one file per family
- `_VARIANTS` with `tiny/small/base`
- `build_*_instance_segmenter(...)`
- `__main__` smoke

Keep the toy semantics distinct:
- `deepmask/sharpmask`: proposal + mask seed/refine
- `cfm/mnc/instancefcn`: proposal/region assembly outputs
- `panet/shapemask/bcnet/refinemask`: ROI mask refinement, bottom-up paths, boundary cues, or coarse-to-fine masks

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "deepmask or sharpmask or cfm or mnc or instancefcn or panet or shapemask or bcnet or refinemask"`

Expected:
- all new classic-family builders pass forward/backward smoke

**Step 5: Commit**

```bash
git add dlhub/vision/instance_segmentation/__init__.py dlhub/vision/instance_segmentation/deepmask.py dlhub/vision/instance_segmentation/sharpmask.py dlhub/vision/instance_segmentation/cfm.py dlhub/vision/instance_segmentation/mnc.py dlhub/vision/instance_segmentation/instancefcn.py dlhub/vision/instance_segmentation/panet.py dlhub/vision/instance_segmentation/shapemask.py dlhub/vision/instance_segmentation/bcnet.py dlhub/vision/instance_segmentation/refinemask.py
git commit -m "feat: add classic instance segmentation families"
```

### Task 5: Implement dense and anchor-free families

**Files:**
- Create: `dlhub/vision/instance_segmentation/sipmask.py`
- Create: `dlhub/vision/instance_segmentation/meinst.py`
- Create: `dlhub/vision/instance_segmentation/orienmask.py`
- Create: `dlhub/vision/instance_segmentation/dct_mask.py`
- Create: `dlhub/vision/instance_segmentation/rtmdet_ins.py`
- Modify: `dlhub/vision/instance_segmentation/__init__.py`

**Step 1: Write the failing test**

Add representative dense-family builders and zoo ids to the red tests.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "sipmask or meinst or orienmask or dct_mask or rtmdet_ins"`

Expected:
- missing imports / unknown builders

**Step 3: Write minimal implementation**

Keep the family distinctions visible:
- `sipmask`: spatial preserve mask coefficients
- `meinst`: mask encoding / latent basis outputs
- `orienmask`: polar/ray geometry outputs
- `dct_mask`: coefficient-domain mask basis outputs
- `rtmdet_ins`: anchor-free decoupled head with dynamic kernels or mask basis fusion

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "sipmask or meinst or orienmask or dct_mask or rtmdet_ins"`

Expected:
- all dense-family builders pass forward/backward smoke

**Step 5: Commit**

```bash
git add dlhub/vision/instance_segmentation/__init__.py dlhub/vision/instance_segmentation/sipmask.py dlhub/vision/instance_segmentation/meinst.py dlhub/vision/instance_segmentation/orienmask.py dlhub/vision/instance_segmentation/dct_mask.py dlhub/vision/instance_segmentation/rtmdet_ins.py
git commit -m "feat: add dense instance segmentation families"
```

### Task 6: Implement query-based families

**Files:**
- Create: `dlhub/vision/instance_segmentation/mask_dino.py`
- Create: `dlhub/vision/instance_segmentation/fastinst.py`
- Create: `dlhub/vision/instance_segmentation/dynamicinst.py`
- Modify: `dlhub/vision/instance_segmentation/__init__.py`

**Step 1: Write the failing test**

Add the new query-family builders to the algorithm smoke test and assert the zoo lists their variants.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "mask_dino or fastinst or dynamicinst"`

Expected:
- missing imports / unknown builders

**Step 3: Write minimal implementation**

Keep the query-style differences explicit:
- `mask_dino`: denoising query branch + iterative box/mask refinement
- `fastinst`: compact query grouping / mask proposal fusion
- `dynamicinst`: dynamic parameter generation per instance query

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "mask_dino or fastinst or dynamicinst"`

Expected:
- all query-family builders pass forward/backward smoke

**Step 5: Commit**

```bash
git add dlhub/vision/instance_segmentation/__init__.py dlhub/vision/instance_segmentation/mask_dino.py dlhub/vision/instance_segmentation/fastinst.py dlhub/vision/instance_segmentation/dynamicinst.py
git commit -m "feat: add query instance segmentation families"
```

### Task 7: Implement contour-based families

**Files:**
- Create: `dlhub/vision/instance_segmentation/deepsnake.py`
- Create: `dlhub/vision/instance_segmentation/e2ec.py`
- Modify: `dlhub/vision/instance_segmentation/__init__.py`

**Step 1: Write the failing test**

Add contour-family builder coverage and zoo id assertions.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "deepsnake or e2ec"`

Expected:
- missing imports / unknown builders

**Step 3: Write minimal implementation**

Represent each family with polygon/contour style outputs:
- contour coordinates or offsets
- rasterized or coarse mask logits
- optional iterative contour refinement tensors

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py -k "deepsnake or e2ec"`

Expected:
- both contour-family builders pass forward/backward smoke

**Step 5: Commit**

```bash
git add dlhub/vision/instance_segmentation/__init__.py dlhub/vision/instance_segmentation/deepsnake.py dlhub/vision/instance_segmentation/e2ec.py
git commit -m "feat: add contour instance segmentation families"
```

### Task 8: Verify the final 40-family instance segmentation zoo

**Files:**
- Verify only

**Step 1: Run focused verification**

Run:
- `pytest -q tests/test_dlhub_vision_instance_segmentation_algorithms.py`
- `pytest -q tests/test_dlhub_vision_instance_segmentation_zoo.py`
- `python scripts/instance_segmentation_zoo.py --list`
- `python scripts/instance_segmentation_zoo.py --smoke dlinst:mask_dino_tiny`

Expected:
- `40` families -> at least `120` local arch ids
- all focused smoke tests pass

**Step 2: Run full verification**

Run: `pytest -q`

Expected:
- full suite passes

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: expand vision instance segmentation zoo to 40 families"
```
