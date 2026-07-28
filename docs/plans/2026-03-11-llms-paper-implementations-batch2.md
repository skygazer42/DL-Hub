# LLM Paper Implementations Batch 2 Plan

**Goal:** Extend `Llms/` with first-class implementations for T5, Flan-T5, and UL2, covering both architecture details and paper-specific training/inference interfaces.

**Architecture:** Implement `T5` as the shared encoder-decoder backbone with relative position bias and shared embeddings. Build `Flan-T5` as an instruction-tuned wrapper over `T5Model` rather than inventing a new architecture. Build `UL2` on top of a T5-compatible backbone, but add Mixture-of-Denoisers mode tags and objective helpers so the paper’s core idea is represented in code and tests.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for T5, Flan-T5, and UL2

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.t5`, `Llms.flan_t5`, `Llms.ul2` exist
- title-case aliases `Llms.T5`, `Llms.Flan_T5`, `Llms.UL2` exist
- `T5Model` is encoder-decoder, uses shared embeddings, relative attention bias, and returns `(B, T, V)`
- `FlanT5Model` wraps a `T5Model` and exposes an instruction-formatting entrypoint
- `UL2Model` exposes mode tags and a denoiser/objective helper that supports `R`, `S`, and `X` modes

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL because the new modules and behaviors do not exist yet.

**Step 3: Write minimal implementation**

Do not implement production code yet beyond what is required to make import failures specific.

**Step 4: Run test to verify it still fails for behavior**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on behavior assertions rather than only missing modules.

**Step 5: Commit**

```bash
git add tests/test_llms_papers.py tests/test_import_regressions.py
git commit -m "test: add batch2 llms paper coverage"
```

### Task 2: Implement T5 encoder-decoder backbone

**Files:**
- Create: `Llms/t5.py`
- Create: `Llms/T5.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- shared encoder/decoder token embeddings
- relative position bias in attention
- no learned absolute positional embedding module
- encoder-decoder forward with `input_ids` and `decoder_input_ids`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on T5-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `T5Config`
- `T5RelativePositionBias`
- `T5Attention`
- `T5EncoderBlock`
- `T5DecoderBlock`
- `T5Model`

Prefer a small, faithful backbone:
- shared token embedding
- relative attention bias
- encoder self-attn + decoder self-attn + decoder cross-attn
- logits projected from decoder hidden states

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: T5 tests PASS.

**Step 5: Commit**

```bash
git add Llms/t5.py Llms/T5.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add t5 paper model"
```

### Task 3: Implement Flan-T5 wrapper

**Files:**
- Create: `Llms/flan_t5.py`
- Create: `Llms/Flan_T5.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- `FlanT5Model` wraps `T5Model`
- instruction-formatting helper prepends a natural-language instruction prefix
- wrapper forward delegates to the base model while preserving output shapes

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Flan-T5-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `FlanT5Config`
- `format_instruction_prompt`
- `FlanT5Model`

Keep scope tight:
- do not reproduce Google’s full task mixture
- expose the key paper idea that Flan-T5 is T5 plus instruction tuning interface

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Flan-T5 tests PASS.

**Step 5: Commit**

```bash
git add Llms/flan_t5.py Llms/Flan_T5.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add flan-t5 wrapper"
```

### Task 4: Implement UL2 mode switching and denoiser helper

**Files:**
- Create: `Llms/ul2.py`
- Create: `Llms/UL2.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- `UL2Model` uses a T5-compatible encoder-decoder base
- model exposes mode tags for `R`, `S`, and `X` denoisers
- objective helper can prepend mode tags and describe denoiser settings
- default tags correspond to the paper-style `[NLU]`, `[S2S]`, `[NLG]` semantics

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on UL2-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `UL2Config`
- `UL2Mode`
- `UL2Objective`
- `UL2Model`

Scope:
- reuse `T5Model`
- encode Mixture-of-Denoisers and mode switching in configuration and helpers
- provide a clean API for formatting UL2-style mode-tagged inputs

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: UL2 tests PASS.

**Step 5: Commit**

```bash
git add Llms/ul2.py Llms/UL2.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add ul2 paper interfaces"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch2 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py
git commit -m "feat: add t5 flan-t5 and ul2 paper modules"
```
