# LLM Paper Implementations Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new root package `Llms/` containing first-batch paper-faithful PyTorch implementations for LLaMA, BLOOM, GPT-NeoX, LoRA, and LLaMA-Adapter.

**Architecture:** Implement these papers as small, importable building blocks rather than full training pipelines. Use one public module per paper with a consistent decoder-only interface where applicable, plus narrowly scoped helpers for paper-specific mechanisms such as ALiBi, partial RoPE, low-rank adapters, and zero-init prompt gating.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing repo conventions

---

### Task 1: Define package layout and regression tests

**Files:**
- Create: `Llms/__init__.py`
- Create: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that import the new package and assert the following:
- `Llms` exports `llama`, `bloom`, `gpt_neox`, `lora`, `llama_adapter`
- each paper module exposes a primary config/model class
- paper-faithful behavior is asserted, not just importability

Include these specific tests:
- `test_llms_package_exports_paper_modules`
- `test_llama_uses_rmsnorm_rope_and_swiglu`
- `test_bloom_uses_alibi_and_embedding_layernorm`
- `test_gpt_neox_uses_partial_rotary_and_parallel_residual`
- `test_lora_linear_zero_init_and_merge_roundtrip`
- `test_llama_adapter_zero_gate_preserves_base_logits_initially`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL with import errors because `Llms/` does not exist yet.

**Step 3: Write minimal implementation**

Create `Llms/__init__.py` so the imports exist, but do not yet implement the paper modules fully.

**Step 4: Run test to verify it still fails for missing behavior**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on missing modules/classes/behavior, confirming the tests are targeting the new feature.

**Step 5: Commit**

```bash
git add Llms/__init__.py tests/test_llms_papers.py
git commit -m "test: add llms paper implementation coverage"
```

### Task 2: Implement LLaMA, BLOOM, and shared decoder primitives

**Files:**
- Create: `Llms/_shared.py`
- Create: `Llms/llama.py`
- Create: `Llms/bloom.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Extend tests to check:
- `LLaMAModel` uses RMSNorm, RoPE on every attention head dimension, SwiGLU MLP, causal decoder flow
- `BloomModel` uses embedding LayerNorm, ALiBi bias in attention scores, GELU MLP, causal decoder flow
- both models return logits of shape `(B, T, vocab_size)`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on LLaMA/BLOOM-specific assertions.

**Step 3: Write minimal implementation**

Implement shared utilities in `Llms/_shared.py`:
- `RMSNorm`
- rotary embedding helpers
- ALiBi slope/bias helpers
- causal decoder attention helpers
- SwiGLU and GELU MLP blocks

Implement `Llms/llama.py`:
- `LLaMAConfig`
- `LLaMAAttention`
- `LLaMABlock`
- `LLaMAModel`

Implement `Llms/bloom.py`:
- `BloomConfig`
- `BloomAttention`
- `BloomBlock`
- `BloomModel`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: LLaMA/BLOOM tests PASS; later GPT-NeoX/LoRA/Adapter tests may still fail.

**Step 5: Commit**

```bash
git add Llms/_shared.py Llms/llama.py Llms/bloom.py tests/test_llms_papers.py
git commit -m "feat: add llama and bloom paper models"
```

### Task 3: Implement GPT-NeoX with paper-specific deviations

**Files:**
- Create: `Llms/gpt_neox.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add/extend tests that verify:
- `GPTNeoXConfig.rotary_pct` defaults to partial RoPE behavior
- only the configured fraction of head dimensions receive RoPE
- block uses parallel residual update, not sequential GPT-2 style update
- model returns logits with causal decoder shape

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL because `Llms/gpt_neox.py` is missing.

**Step 3: Write minimal implementation**

Implement:
- `GPTNeoXConfig`
- `GPTNeoXAttention`
- `GPTNeoXParallelBlock`
- `GPTNeoXModel`

Reuse shared decoder helpers where possible, but keep GPT-NeoX-specific logic local:
- partial rotary application via `rotary_pct`
- parallel residual combining attention and MLP updates from the same normalized hidden state

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: GPT-NeoX tests PASS; LoRA/Adapter tests may still fail.

**Step 5: Commit**

```bash
git add Llms/gpt_neox.py tests/test_llms_papers.py
git commit -m "feat: add gpt-neox paper model"
```

### Task 4: Implement LoRA adapters and merge semantics

**Files:**
- Create: `Llms/lora.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests covering:
- `LoRALinear` freezes base linear weights by default
- low-rank path uses `B @ A`-equivalent decomposition with zero-init behavior
- forward before training matches base linear output because the delta path is initially zero
- `merge()` and `unmerge()` preserve outputs round-trip

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL because `LoRALinear` does not exist.

**Step 3: Write minimal implementation**

Implement:
- `LoRAConfig`
- `LoRALinear`
- optional helper `apply_lora_to_attention_projections`

Keep the first version focused on linear projection adaptation rather than whole-model surgery.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: LoRA tests PASS; Adapter tests may still fail.

**Step 5: Commit**

```bash
git add Llms/lora.py tests/test_llms_papers.py
git commit -m "feat: add lora paper module"
```

### Task 5: Implement LLaMA-Adapter prompt-gating wrapper

**Files:**
- Create: `Llms/llama_adapter.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- adapter prompts are inserted into the topmost `L` transformer layers only
- prompt tensors are learnable
- zero-initialized gate makes the adapter path preserve base logits initially
- increasing the gate changes outputs while keeping tensor shapes valid

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL because the adapter wrapper does not exist.

**Step 3: Write minimal implementation**

Implement:
- `LLaMAAdapterConfig`
- `ZeroInitPromptGate`
- `LLaMAAdapterModel`

Scope for this repo:
- wrap `LLaMAModel`
- prepend learnable prompts at selected top layers
- use zero-init gated prompt attention contribution
- keep the implementation text-only; document multimodal extension as future work rather than adding image encoders now

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: All tests in `tests/test_llms_papers.py` PASS.

**Step 5: Commit**

```bash
git add Llms/llama_adapter.py tests/test_llms_papers.py
git commit -m "feat: add llama-adapter paper module"
```

### Task 6: Final verification and import regression coverage

**Files:**
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add import coverage for the new `Llms` package in `tests/test_import_regressions.py`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_import_regressions.py::test_imports -q`
Expected: FAIL if the new package is not included or not import-safe.

**Step 3: Write minimal implementation**

Update import-regression coverage to include:
- `Llms`
- `Llms.llama`
- `Llms.bloom`
- `Llms.gpt_neox`
- `Llms.lora`
- `Llms.llama_adapter`

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/test_import_regressions.py
git commit -m "test: cover llms paper package imports"
```
