# Vision Co-Segmentation Implementation Plan

**Goal:** Build a local, compact-first vision co-segmentation zoo with six algorithm families, AST-based discovery, CLI utilities, and smoke coverage.

**Architecture:** Create a dedicated `dlhub/vision/co_segmentation/` package with one file per family plus a shared `_common.py` for image-set validation, flatten/unflatten helpers, tiny encoders, group fusion, and logits heads. Add an AST-discovered `co_segmentation_zoo.py`, a matching CLI, and a focused pytest module that drives TDD for listing, building, and `(B,T,C,H,W)` forward smoke across all tiny variants.

**Tech Stack:** Python, PyTorch, pytest, subprocess CLI smoke tests, AST-based local zoo discovery.

---

### Task 1: Add failing zoo-level co-segmentation tests

**Files:**
- Create: `F:\DL-Hub\tests\test_dlhub_vision_co_segmentation_zoo.py`

**Step 1: Write the failing test**

```python
def test_co_segmentation_zoo_lists_families() -> None:
    from dlhub.vision.co_segmentation_zoo import list_local_arches

    arches = list_local_arches()
    assert len(arches) >= 18
    assert "coseg:siamese_coseg_tiny" in arches
    assert "coseg:transformer_coseg_base" in arches
```

Add a second test that parametrizes all six `*_tiny` variants and asserts:
- the builder returns a dict
- `logits.shape == (2, 3, 2, 64, 64)`
- `masks.shape == (2, 3, 64, 64)`

Add a subprocess test for:
- `python scripts/co_segmentation_zoo.py --list --limit 8`
- `python scripts/co_segmentation_zoo.py --smoke coseg:siamese_coseg_tiny`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL because `dlhub.vision.co_segmentation_zoo` does not exist yet.

**Step 3: Write minimal implementation**

Create only the minimum skeleton needed for imports to resolve later:

```python
# dlhub/vision/co_segmentation_zoo.py
def list_local_arches():
    return []
```

Do not add family implementations yet.

**Step 4: Run test to verify it still fails for the right reason**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL on missing arches, not import errors.

**Step 5: Commit**

```bash
git add tests/test_dlhub_vision_co_segmentation_zoo.py dlhub/vision/co_segmentation_zoo.py
git commit -m "test: add co-segmentation zoo coverage"
```

### Task 2: Add the package skeleton and shared co-segmentation primitives

**Files:**
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\__init__.py`
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\_common.py`

**Step 1: Write the failing test**

Extend `tests/test_dlhub_vision_co_segmentation_zoo.py` with a direct `_common.py` smoke:

```python
def test_flatten_group_roundtrip() -> None:
    from dlhub.vision.co_segmentation._common import flatten_group, unflatten_group

    x = torch.randn(2, 3, 4, 8, 8)
    flat = flatten_group(x)
    y = unflatten_group(flat, batch=2, set_size=3)
    assert tuple(flat.shape) == (6, 4, 8, 8)
    assert torch.allclose(x, y)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL because `_common.py` and helpers do not exist yet.

**Step 3: Write minimal implementation**

Implement:
- `check_btchw(images)`
- `logits_to_masks(logits)`
- `flatten_group(images)`
- `unflatten_group(images, *, batch, set_size)`
- `TinyCoSegEncoder`
- `GroupFusionBlock`
- `CoSegHead`

Keep all helpers pure-torch, small, and CPU-friendly.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: the roundtrip helper test passes; zoo tests still fail.

**Step 5: Commit**

```bash
git add dlhub/vision/co_segmentation/__init__.py dlhub/vision/co_segmentation/_common.py tests/test_dlhub_vision_co_segmentation_zoo.py
git commit -m "feat: add co-segmentation shared primitives"
```

### Task 3: Add AST-discovered zoo wiring and CLI

**Files:**
- Modify: `F:\DL-Hub\dlhub\vision\co_segmentation_zoo.py`
- Create: `F:\DL-Hub\scripts\co_segmentation_zoo.py`

**Step 1: Write the failing test**

Add a CLI search assertion:

```python
assert "Co-segmentation local zoo" in list_proc.stdout
assert "total_arches=" in list_proc.stdout
```

Add a direct zoo build assertion using an eventually valid arch id like `coseg:siamese_coseg_tiny`.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL because the registry cannot discover family files yet.

**Step 3: Write minimal implementation**

Implement `co_segmentation_zoo.py` with the same pattern used by:
- `F:\DL-Hub\dlhub\vision\face_parsing_zoo.py`
- `F:\DL-Hub\dlhub\vision\video_summarization_zoo.py`

Implement `scripts/co_segmentation_zoo.py` with:
- `--list`
- `--search`
- `--limit`
- `--smoke`
- `--batch-size`
- `--set-size`
- `--image-size`
- `--in-channels`
- `--num-classes`
- `--width-mult`
- `--dropout`

**Step 4: Run test to verify the failure moves forward**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL only because no family modules exist yet.

**Step 5: Commit**

```bash
git add dlhub/vision/co_segmentation_zoo.py scripts/co_segmentation_zoo.py tests/test_dlhub_vision_co_segmentation_zoo.py
git commit -m "feat: add co-segmentation zoo and CLI"
```

### Task 4: Implement the first three co-segmentation families

**Files:**
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\siamese_coseg.py`
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\group_proto_net.py`
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\co_attention_fpn.py`

**Step 1: Write the failing test**

In `tests/test_dlhub_vision_co_segmentation_zoo.py`, require:

```python
assert "coseg:siamese_coseg_tiny" in arches
assert "coseg:group_proto_net_small" in arches
assert "coseg:co_attention_fpn_base" in arches
```

Parametrize smoke coverage with:
- `coseg:siamese_coseg_tiny`
- `coseg:group_proto_net_tiny`
- `coseg:co_attention_fpn_tiny`

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL with unknown arch errors for those three families.

**Step 3: Write minimal implementation**

For each family:
- define `_VARIANTS` as a literal dict
- expose `build_<family>_co_segmentor(...)`
- return a dict with `logits` and `masks`
- optionally add family-specific outputs such as `group_tokens`, `prototype_masks`, or `co_attention`

**Step 4: Run test to verify it passes for those families**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q -k "siamese_coseg or group_proto_net or co_attention_fpn or lists_families or script"`

Expected: PASS for the implemented families, remaining failures only for the missing families.

**Step 5: Commit**

```bash
git add dlhub/vision/co_segmentation/siamese_coseg.py dlhub/vision/co_segmentation/group_proto_net.py dlhub/vision/co_segmentation/co_attention_fpn.py tests/test_dlhub_vision_co_segmentation_zoo.py
git commit -m "feat: add core co-segmentation families"
```

### Task 5: Implement the remaining three co-segmentation families

**Files:**
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\cosal_uformer.py`
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\transformer_coseg.py`
- Create: `F:\DL-Hub\dlhub\vision\co_segmentation\consensus_refiner.py`

**Step 1: Write the failing test**

In `tests/test_dlhub_vision_co_segmentation_zoo.py`, require:

```python
assert "coseg:cosal_uformer_tiny" in arches
assert "coseg:transformer_coseg_small" in arches
assert "coseg:consensus_refiner_base" in arches
```

Parametrize smoke coverage with:
- `coseg:cosal_uformer_tiny`
- `coseg:transformer_coseg_tiny`
- `coseg:consensus_refiner_tiny`

Raise the minimum family count expectation to `18` arches if it was temporarily lower.

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: FAIL with unknown arch errors for the three missing families.

**Step 3: Write minimal implementation**

Implement:
- `cosal_uformer`: U-shaped shared encoder/decoder with group consensus injection
- `transformer_coseg`: patch tokens plus group token exchange
- `consensus_refiner`: coarse masks plus consensus refinement residual

All families must accept `(B, T, C, H, W)` and upsample back to full image resolution.

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`

Expected: PASS

**Step 5: Commit**

```bash
git add dlhub/vision/co_segmentation/cosal_uformer.py dlhub/vision/co_segmentation/transformer_coseg.py dlhub/vision/co_segmentation/consensus_refiner.py tests/test_dlhub_vision_co_segmentation_zoo.py
git commit -m "feat: complete co-segmentation zoo families"
```

### Task 6: Verify CLI behavior and document the final coverage

**Files:**
- Modify: `F:\DL-Hub\docs\plans\2026-03-19-vision-co-segmentation-design.md`

**Step 1: Write the failing test**

No new pytest code. The verification target is command-level coverage.

**Step 2: Run verification commands**

Run:
- `python -m pytest tests\test_dlhub_vision_co_segmentation_zoo.py -q`
- `python scripts\co_segmentation_zoo.py --list --limit 15`
- `python scripts\co_segmentation_zoo.py --search transformer_coseg --list --limit 20`
- `python scripts\co_segmentation_zoo.py --smoke coseg:siamese_coseg_tiny --image-size 64 --set-size 3 --num-classes 2 --width-mult 0.5`
- `python scripts\co_segmentation_zoo.py --smoke coseg:transformer_coseg_tiny --image-size 64 --set-size 3 --num-classes 2 --width-mult 0.5`
- `python scripts\co_segmentation_zoo.py --smoke coseg:consensus_refiner_tiny --image-size 64 --set-size 3 --num-classes 2 --width-mult 0.5`

Expected:
- pytest passes
- CLI lists `total_arches=18`
- search output isolates the requested family
- all smoke commands exit `0`

**Step 3: Write minimal documentation update**

Update `F:\DL-Hub\docs\plans\2026-03-19-vision-co-segmentation-design.md` with:
- the final family list
- the final arch count

**Step 4: Run verification again**

Repeat the verification commands above after the docs update.

**Step 5: Commit**

```bash
git add docs/plans/2026-03-19-vision-co-segmentation-design.md
git commit -m "docs: finalize co-segmentation design coverage"
```
