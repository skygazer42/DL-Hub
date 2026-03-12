# LLM Paper Implementations Batch 6 Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `Llms/` with paper-shaped implementations for ScienceQA, Dolly, and GPT4All, covering multimodal CoT question answering, human-generated instruction finetuning, and large-scale distilled assistant training.

**Architecture:** Implement `ScienceQA` as a multimodal QA wrapper with rationale-aware prompt formatting and answer scoring rather than a full benchmark dataset dump. Implement `Dolly` as a Pythia-based instruction-following wrapper with the paper’s prompt sections and human-generated `dolly-15k` metadata. Implement `GPT4All` as a distilled assistant-training wrapper over a LLaMA-like backbone, exposing data curation metadata, LoRA-style finetuning strategy, and CPU-friendly quantization metadata.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for ScienceQA, Dolly, and GPT4All

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.scienceqa`, `Llms.dolly`, and `Llms.gpt4all` exist
- title-case aliases `Llms.ScienceQA`, `Llms.Dolly`, and `Llms.GPT4All` exist
- `ScienceQA` exposes multimodal CoT prompt building and answer scoring
- `Dolly` exposes Databricks-style prompt sections and a Pythia-based wrapper
- `GPT4All` exposes GPT-3.5-Turbo distillation metadata, LoRA strategy, and quantized deployment metadata

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL because the new modules and behaviors do not exist yet.

**Step 3: Write minimal implementation**

Do not implement production code yet beyond what is required to make the failures specific.

**Step 4: Run test to verify it still fails for behavior**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on behavior assertions rather than only missing modules.

**Step 5: Commit**

```bash
git add tests/test_llms_papers.py tests/test_import_regressions.py
git commit -m "test: add batch6 llms paper coverage"
```

### Task 2: Implement ScienceQA multimodal CoT wrapper

**Files:**
- Create: `Llms/scienceqa.py`
- Create: `Llms/ScienceQA.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- prompt builder includes question, choices, lecture, and step-by-step reasoning cue
- multimodal example metadata distinguishes image/text context
- model returns answer logits with expected shape

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on ScienceQA-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `ScienceQAExample`
- `ScienceQAConfig`
- `format_scienceqa_prompt`
- `ScienceQAModel`

Keep scope tight:
- encode the paper’s rationale-rich prompt and multimodal flagging
- use a compact answer-scoring model instead of reproducing GPT-3

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: ScienceQA tests PASS.

**Step 5: Commit**

```bash
git add Llms/scienceqa.py Llms/ScienceQA.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add scienceqa paper module"
```

### Task 3: Implement Dolly instruction-tuned wrapper

**Files:**
- Create: `Llms/dolly.py`
- Create: `Llms/Dolly.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- prompt format uses `Instruction` / `Input` / `Response` sections
- metadata reflects `databricks-dolly-15k` and human-generated data
- model wraps a `PythiaModel`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Dolly-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `DollyExample`
- `DollyConfig`
- `format_dolly_prompt`
- `DollyModel`

Scope:
- model Dolly 2.0’s Pythia-based instruction tuning interface
- keep the implementation lean and metadata-forward

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Dolly tests PASS.

**Step 5: Commit**

```bash
git add Llms/dolly.py Llms/Dolly.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add dolly paper module"
```

### Task 4: Implement GPT4All distilled assistant wrapper

**Files:**
- Create: `Llms/gpt4all.py`
- Create: `Llms/GPT4All.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- data curation metadata tracks initial/cleaned/final pair counts
- model identifies GPT-3.5-Turbo distillation and LoRA finetuning
- quantization metadata exposes 4-bit CPU deployment

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on GPT4All-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `GPT4AllDataCuration`
- `GPT4AllConfig`
- `format_gpt4all_prompt`
- `GPT4AllModel`

Scope:
- wrap a LLaMA-like backbone
- expose the paper’s distilled assistant training pipeline rather than another new decoder design

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: GPT4All tests PASS.

**Step 5: Commit**

```bash
git add Llms/gpt4all.py Llms/GPT4All.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add gpt4all paper module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch6 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch6.md
git commit -m "feat: add scienceqa dolly and gpt4all paper modules"
```
