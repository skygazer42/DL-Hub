# Vision FGVC 40-Family Implementation Plan

**Goal:** Add a new image-level fine-grained visual recognition task to DL-Hub with 40 distinct algorithm families, a discoverable local zoo, and pure-torch compact-first classifiers.

**Architecture:** Create a dedicated `dlhub/vision/fine_grained_recognition/` task directory instead of mixing FGVC models into generic backbones. Keep the existing zoo conventions: one family per file, `_VARIANTS` for `tiny/small/base`, a `build_*_fgvc_classifier(...)` factory, and a `__main__` smoke block. Add a lazy `fine_grained_recognition_zoo` plus a CLI script so all 120 variants are enumerable and buildable by id.

**Tech Stack:** Python, PyTorch, pytest, AST-based lazy model discovery, repo-local compact-first CNN/part-attention/transformer utilities.

---

## Target Family Inventory (40)

- Bilinear / covariance: `bilinear_cnn`, `compact_bilinear`, `kernel_pooling`, `lowrank_bilinear`, `hierarchical_bilinear`, `isqrt_cov`, `mpn_cov`, `ws_ban`
- Part / localization: `part_rcnn`, `partnet`, `part_stacked_cnn`, `pa_cnn`, `racnn`, `ma_cnn`, `dfl_cnn`, `nts_net`, `tasn`, `s3n`, `mge_cnn`, `pmg`
- Relation / prototype / multi-granularity: `osme_mamc`, `api_net`, `crossx`, `region_grouping`, `dcl`, `ws_dan`, `proto_pnet`, `hse`, `interp_parts`, `ga_cnn`
- Transformer / modern FGVC: `transfg`, `ffvt`, `pedtrans`, `vit_fod`, `aftrans`, `sim_trans`, `pca_net`, `metaformer_fgvc`, `pim`, `cvl`

### Task 1: Add failing FGVC tests and convention coverage

**Files:**
- Create: `tests/test_dlhub_vision_fine_grained_recognition_algorithms.py`
- Create: `tests/test_dlhub_vision_fine_grained_recognition_zoo.py`
- Modify: `tests/test_zoo_conventions_smoke.py`

**Step 1: Write the failing test**

Add an algorithms smoke test that imports `dlhub.vision.fine_grained_recognition` and runs forward/backward on all 40 `*_tiny` builders. Output contract:
- `dict`
- must include `logits`
- `logits.shape == (B, num_classes)`

Add a zoo test that:
- imports `dlhub.vision.fine_grained_recognition_zoo`
- asserts `len(list_local_arches()) >= 120`
- asserts representative ids exist:
  - `dlfgvc:bilinear_cnn_tiny`
  - `dlfgvc:racnn_tiny`
  - `dlfgvc:ws_dan_tiny`
  - `dlfgvc:transfg_tiny`
  - `dlfgvc:metaformer_fgvc_tiny`
- parametrizes over all `dlfgvc:*_tiny` ids and runs forward/backward smoke
- builds representative `small/base` variants
- subprocess-smokes `scripts/fine_grained_recognition_zoo.py --list` and `--smoke`

Extend `tests/test_zoo_conventions_smoke.py` so the new FGVC directory is covered by `_VARIANTS` / builder / `__main__` checks.

**Step 2: Run test to verify it fails**

Run:
- `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py`
- `pytest -q tests/test_dlhub_vision_fine_grained_recognition_zoo.py`

Expected:
- import errors for missing task directory / zoo module / script

**Step 3: Commit**

```bash
git add tests/test_dlhub_vision_fine_grained_recognition_algorithms.py tests/test_dlhub_vision_fine_grained_recognition_zoo.py tests/test_zoo_conventions_smoke.py
git commit -m "test: add fgvc task coverage"
```

### Task 2: Add FGVC common blocks and local zoo wiring

**Files:**
- Create: `dlhub/vision/fine_grained_recognition/_common.py`
- Create: `dlhub/vision/fine_grained_recognition/__init__.py`
- Create: `dlhub/vision/fine_grained_recognition_zoo.py`
- Create: `scripts/fine_grained_recognition_zoo.py`

**Step 1: Write the failing test**

Use Task 1's red tests. Do not write family files yet.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_zoo.py::test_fgvc_zoo_lists_120_plus_arches`

Expected:
- missing import or missing `list_local_arches/build_local_model`

**Step 3: Write minimal implementation**

Add shared compact-first primitives:
- NCHW validation
- compact CNN feature extractor
- part-attention pooling
- bilinear / covariance-style pooling helpers
- lightweight patch-token encoder for transformer-style FGVC
- simple prototype and relation heads

Add zoo discovery using AST:
- extract `_VARIANTS`
- extract `build_*_fgvc_classifier`
- lazy import builders
- prefix `dlfgvc:`

Add CLI script:
- `--list`
- `--search`
- `--smoke`

**Step 4: Run test to verify it now fails for missing families, not missing infrastructure**

Run:
- `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py`
- `pytest -q tests/test_dlhub_vision_fine_grained_recognition_zoo.py`

**Step 5: Commit**

```bash
git add dlhub/vision/fine_grained_recognition/_common.py dlhub/vision/fine_grained_recognition/__init__.py dlhub/vision/fine_grained_recognition_zoo.py scripts/fine_grained_recognition_zoo.py
git commit -m "feat: add fgvc task scaffolding"
```

### Task 3: Implement bilinear and covariance families

**Files:**
- Create: `dlhub/vision/fine_grained_recognition/bilinear_cnn.py`
- Create: `dlhub/vision/fine_grained_recognition/compact_bilinear.py`
- Create: `dlhub/vision/fine_grained_recognition/kernel_pooling.py`
- Create: `dlhub/vision/fine_grained_recognition/lowrank_bilinear.py`
- Create: `dlhub/vision/fine_grained_recognition/hierarchical_bilinear.py`
- Create: `dlhub/vision/fine_grained_recognition/isqrt_cov.py`
- Create: `dlhub/vision/fine_grained_recognition/mpn_cov.py`
- Create: `dlhub/vision/fine_grained_recognition/ws_ban.py`
- Modify: `dlhub/vision/fine_grained_recognition/__init__.py`

**Step 1: Write the failing test**

Ensure all eight builders are listed in the algorithms smoke test and representative zoo ids are asserted.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "bilinear or compact or kernel or lowrank or hierarchical or isqrt or mpn or ws_ban"`

**Step 3: Write minimal implementation**

Use one shared bilinear base with family-specific extra outputs:
- bilinear embedding / sketch features / kernel response / low-rank factors / hierarchical pooled features / covariance descriptors / attention maps

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "bilinear or compact or kernel or lowrank or hierarchical or isqrt or mpn or ws_ban"`

**Step 5: Commit**

```bash
git add dlhub/vision/fine_grained_recognition/__init__.py dlhub/vision/fine_grained_recognition/bilinear_cnn.py dlhub/vision/fine_grained_recognition/compact_bilinear.py dlhub/vision/fine_grained_recognition/kernel_pooling.py dlhub/vision/fine_grained_recognition/lowrank_bilinear.py dlhub/vision/fine_grained_recognition/hierarchical_bilinear.py dlhub/vision/fine_grained_recognition/isqrt_cov.py dlhub/vision/fine_grained_recognition/mpn_cov.py dlhub/vision/fine_grained_recognition/ws_ban.py
git commit -m "feat: add bilinear fgvc families"
```

### Task 4: Implement part and localization families

**Files:**
- Create: `dlhub/vision/fine_grained_recognition/part_rcnn.py`
- Create: `dlhub/vision/fine_grained_recognition/partnet.py`
- Create: `dlhub/vision/fine_grained_recognition/part_stacked_cnn.py`
- Create: `dlhub/vision/fine_grained_recognition/pa_cnn.py`
- Create: `dlhub/vision/fine_grained_recognition/racnn.py`
- Create: `dlhub/vision/fine_grained_recognition/ma_cnn.py`
- Create: `dlhub/vision/fine_grained_recognition/dfl_cnn.py`
- Create: `dlhub/vision/fine_grained_recognition/nts_net.py`
- Create: `dlhub/vision/fine_grained_recognition/tasn.py`
- Create: `dlhub/vision/fine_grained_recognition/s3n.py`
- Create: `dlhub/vision/fine_grained_recognition/mge_cnn.py`
- Create: `dlhub/vision/fine_grained_recognition/pmg.py`
- Modify: `dlhub/vision/fine_grained_recognition/__init__.py`

**Step 1: Write the failing test**

Keep the new builders in the algorithms smoke test and representative ids in the zoo test.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "part_rcnn or partnet or part_stacked or pa_cnn or racnn or ma_cnn or dfl_cnn or nts_net or tasn or s3n or mge_cnn or pmg"`

**Step 3: Write minimal implementation**

Use a shared part-based base with family-specific cues:
- proposal boxes
- part attention maps
- sequential glimpses
- multi-attention logits
- part filters
- navigator scores
- trilinear sampling logits
- scale snapshots
- multi-granularity embeddings

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "part_rcnn or partnet or part_stacked or pa_cnn or racnn or ma_cnn or dfl_cnn or nts_net or tasn or s3n or mge_cnn or pmg"`

### Task 5: Implement relation, prototype, and complementary-learning families

**Files:**
- Create: `dlhub/vision/fine_grained_recognition/osme_mamc.py`
- Create: `dlhub/vision/fine_grained_recognition/api_net.py`
- Create: `dlhub/vision/fine_grained_recognition/crossx.py`
- Create: `dlhub/vision/fine_grained_recognition/region_grouping.py`
- Create: `dlhub/vision/fine_grained_recognition/dcl.py`
- Create: `dlhub/vision/fine_grained_recognition/ws_dan.py`
- Create: `dlhub/vision/fine_grained_recognition/proto_pnet.py`
- Create: `dlhub/vision/fine_grained_recognition/hse.py`
- Create: `dlhub/vision/fine_grained_recognition/interp_parts.py`
- Create: `dlhub/vision/fine_grained_recognition/ga_cnn.py`
- Modify: `dlhub/vision/fine_grained_recognition/__init__.py`

**Step 1: Write the failing test**

Use the algorithms smoke list and zoo id assertions as red tests.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "osme_mamc or api_net or crossx or region_grouping or dcl or ws_dan or proto_pnet or hse or interp_parts or ga_cnn"`

**Step 3: Write minimal implementation**

Use shared relation/prototype heads with family-specific extras:
- mutual attention logits
- pair interaction scores
- complementary region outputs
- region-group assignments
- destruction/reconstruction logits
- augmented attention maps
- prototype activations
- hierarchical logits
- interpretable part scores
- granularity-stage features

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "osme_mamc or api_net or crossx or region_grouping or dcl or ws_dan or proto_pnet or hse or interp_parts or ga_cnn"`

### Task 6: Implement transformer and modern FGVC families

**Files:**
- Create: `dlhub/vision/fine_grained_recognition/transfg.py`
- Create: `dlhub/vision/fine_grained_recognition/ffvt.py`
- Create: `dlhub/vision/fine_grained_recognition/pedtrans.py`
- Create: `dlhub/vision/fine_grained_recognition/vit_fod.py`
- Create: `dlhub/vision/fine_grained_recognition/aftrans.py`
- Create: `dlhub/vision/fine_grained_recognition/sim_trans.py`
- Create: `dlhub/vision/fine_grained_recognition/pca_net.py`
- Create: `dlhub/vision/fine_grained_recognition/metaformer_fgvc.py`
- Create: `dlhub/vision/fine_grained_recognition/pim.py`
- Create: `dlhub/vision/fine_grained_recognition/cvl.py`
- Modify: `dlhub/vision/fine_grained_recognition/__init__.py`

**Step 1: Write the failing test**

Keep all transformer-family builders in the red smoke list.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "transfg or ffvt or pedtrans or vit_fod or aftrans or sim_trans or pca_net or metaformer_fgvc or pim or cvl"`

**Step 3: Write minimal implementation**

Use a shared lightweight patch-token encoder with family-specific extras:
- selected token indices
- part-enhanced cls tokens
- pose/meta tokens
- object-difference tokens
- multi-scale attention fusion
- structure relations
- co-attention maps
- meta-information fusion
- plug-in region masks
- language-conditioned auxiliary embeddings

**Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py -k "transfg or ffvt or pedtrans or vit_fod or aftrans or sim_trans or pca_net or metaformer_fgvc or pim or cvl"`

### Task 7: Final verification

**Files:**
- Verify only

**Step 1: Run focused verification**

Run:
- `pytest -q tests/test_dlhub_vision_fine_grained_recognition_algorithms.py`
- `pytest -q tests/test_dlhub_vision_fine_grained_recognition_zoo.py`
- `pytest -q tests/test_zoo_conventions_smoke.py`
- `python scripts/fine_grained_recognition_zoo.py --list --limit 12`
- `python scripts/fine_grained_recognition_zoo.py --smoke dlfgvc:transfg_tiny`

Expected:
- `120` total arch ids
- all focused smoke tests pass

**Step 2: Run full verification**

Run: `pytest -q`

Expected:
- full suite passes

**Step 3: Commit**

```bash
git add -A
git commit -m "feat: add fgvc task with 40 families"
```
