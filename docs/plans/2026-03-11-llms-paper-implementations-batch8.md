# LLM Paper Implementations Batch 8 Plan

**Goal:** Extend `Llms/` with paper-shaped implementations for Segment Anything, BLIP, and InstructBLIP, covering promptable segmentation, unified vision-language pretraining, and instruction-tuned multimodal prompting.

**Architecture:** Implement `Segment Anything` as a promptable segmentation abstraction with an image encoder, prompt encoder, lightweight mask decoder, and SA-1B-style automatic mask generation metadata. Implement `BLIP` as a unified image-language wrapper with separate image/text encoders, a cross-attentive multimodal encoder, a decoder-style language head, and a compact CapFilt data curation helper. Implement `InstructBLIP` as a BLIP-2-style multimodal wrapper where instructions condition the query transformer and visual prefix prompts supplied to a frozen language model.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for Segment Anything, BLIP, and InstructBLIP

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.segment_anything`, `Llms.blip`, and `Llms.instructblip` exist
- title-case aliases `Llms.Segment_Anything`, `Llms.BLIP`, and `Llms.InstructBLIP` exist
- `Segment Anything` exposes prompt encoding, multiple-mask decoding, and SA-1B data engine metadata
- `BLIP` exposes image/text/multimodal components plus CapFilt-style caption filtering
- `InstructBLIP` exposes instruction-conditioned query extraction and a frozen language-model wrapper

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL because the new modules and exports do not exist yet.

**Step 3: Write minimal implementation**

Do not write production code yet beyond what is necessary to make the failure specific.

**Step 4: Run test to verify it still fails for behavior**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on missing behavior assertions rather than only missing imports.

**Step 5: Commit**

```bash
git add tests/test_llms_papers.py tests/test_import_regressions.py
git commit -m "test: add batch8 llms paper coverage"
```

### Task 2: Implement Segment Anything promptable segmentation abstraction

**Files:**
- Create: `Llms/segment_anything.py`
- Create: `Llms/Segment_Anything.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- prompt objects support point, box, and mask modes
- model returns multiple candidate masks and IoU/confidence scores
- automatic mask generation uses a grid-like prompting schedule and SA-1B metadata

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Segment Anything-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `SAMPrompt`
- `SAMConfig`
- `SAMDataEngine`
- `SAMAutomaticMaskGenerator`
- `SegmentAnythingModel`

Keep scope tight:
- model promptable segmentation, ambiguity-aware multi-mask output, and reusable image embeddings
- avoid reproducing the full Meta training system

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Segment Anything tests PASS.

**Step 5: Commit**

```bash
git add Llms/segment_anything.py Llms/Segment_Anything.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add segment anything paper module"
```

### Task 3: Implement BLIP unified vision-language wrapper

**Files:**
- Create: `Llms/blip.py`
- Create: `Llms/BLIP.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- model exposes image encoder, text encoder, multimodal encoder, and decoder-style generation head
- CapFilt curation can score and filter noisy captions
- forward pass returns ITC, ITM, and LM-shaped outputs

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on BLIP-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `BLIPConfig`
- `CapFiltPair`
- `CapFiltPipeline`
- `BLIPModel`

Scope:
- encode BLIP’s unified multi-task framing and web-caption cleanup
- keep the architecture compact and repo-consistent

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: BLIP tests PASS.

**Step 5: Commit**

```bash
git add Llms/blip.py Llms/BLIP.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add blip paper module"
```

### Task 4: Implement InstructBLIP instruction-aware multimodal wrapper

**Files:**
- Create: `Llms/instructblip.py`
- Create: `Llms/InstructBLIP.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- instruction tokens condition the query extraction path
- visual prefix prompts are supplied before textual instructions
- model wraps a BLIP-2-style image encoder and query transformer with instruction-aware behavior

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on InstructBLIP-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `InstructBLIPConfig`
- `InstructionAwareQFormer`
- `InstructBLIPModel`
- `format_instructblip_prompt`

Scope:
- distinguish it from plain `BLIP2` by letting instructions guide query features
- keep the wrapper lean and metadata-forward

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: InstructBLIP tests PASS.

**Step 5: Commit**

```bash
git add Llms/instructblip.py Llms/InstructBLIP.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add instructblip paper module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch8 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring and keep behavior minimal.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch8.md
git commit -m "feat: add segment anything blip and instructblip paper modules"
```
