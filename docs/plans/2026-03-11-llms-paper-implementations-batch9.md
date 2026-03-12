# LLM Paper Implementations Batch 9 Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `Llms/` with paper-shaped implementations for RedPajama, StarCoder, and the Prompt Engineering Guide, covering open-data reproduction, code-specialized long-context decoding, and structured prompting strategies.

**Architecture:** Implement `RedPajama` as a reproducible corpus-mixture abstraction with seven canonical slices, token allocation, and quality-filter metadata rather than a backbone network. Implement `StarCoder` as a code-LLM wrapper with permissively licensed GitHub-centric data metadata, fill-in-the-middle prompting, long-context settings, and assistant-style usage. Implement `Prompt Engineering Guide` as a prompt-construction toolkit exposing instruction/context/input/output formatting, few-shot prompt assembly, and sampling profiles for factual vs creative tasks.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for RedPajama, StarCoder, and Prompt Engineering Guide

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.red_pajama`, `Llms.starcoder`, and `Llms.prompt_engineering_guide` exist
- title-case aliases `Llms.RedPajama`, `Llms.StarCoder`, and `Llms.Prompt_Engineering_Guide` exist
- `RedPajama` exposes the seven-slice data mixture and token allocation behavior
- `StarCoder` exposes long-context code-assistant behavior, fill-in-the-middle formatting, and open-license metadata
- `Prompt Engineering Guide` exposes structured prompt elements, few-shot assembly, and task-specific sampling recommendations

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
git commit -m "test: add batch9 llms paper coverage"
```

### Task 2: Implement RedPajama open-data mixture abstraction

**Files:**
- Create: `Llms/red_pajama.py`
- Create: `Llms/RedPajama.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- canonical slices include CommonCrawl, C4, GitHub, arXiv, Books, Wikipedia, and StackExchange
- token allocation sums correctly
- dataset metadata reflects license filtering and large-scale reproduction goals

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on RedPajama-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `RedPajamaSlice`
- `RedPajamaConfig`
- `canonical_red_pajama_slices`
- `RedPajamaDataset`

Keep scope tight:
- represent the data recipe, filtering, and token budgeting
- avoid pretending the note specifies a new decoder architecture

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: RedPajama tests PASS.

**Step 5: Commit**

```bash
git add Llms/red_pajama.py Llms/RedPajama.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add redpajama paper module"
```

### Task 3: Implement StarCoder code-LLM wrapper

**Files:**
- Create: `Llms/starcoder.py`
- Create: `Llms/StarCoder.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- fill-in-the-middle prompt formatting exists
- metadata reflects 80+ languages, 8k context, OpenRAIL licensing, and GitHub-derived training data
- model exposes code-assistant-style logits

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on StarCoder-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `StarCoderDataConfig`
- `StarCoderConfig`
- `format_fim_prompt`
- `StarCoderModel`

Scope:
- capture the paper’s code-specialized long-context assistant framing
- keep the architecture compact and repo-consistent

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: StarCoder tests PASS.

**Step 5: Commit**

```bash
git add Llms/starcoder.py Llms/StarCoder.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add starcoder paper module"
```

### Task 4: Implement Prompt Engineering Guide toolkit

**Files:**
- Create: `Llms/prompt_engineering_guide.py`
- Create: `Llms/Prompt_Engineering_Guide.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- structured prompts include instruction, context, input, and output indicator blocks
- few-shot prompts can render demonstrations
- sampling recommendations differentiate factual and creative tasks by temperature and top-p

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on prompt-engineering-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `PromptEngineeringConfig`
- `PromptExample`
- `PromptTemplate`
- `PromptEngineeringGuide`

Scope:
- encode the guide’s prompt elements and task-aware sampling advice
- avoid duplicating the dedicated `Chain_of_Thought` module beyond lightweight strategy references

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Prompt Engineering Guide tests PASS.

**Step 5: Commit**

```bash
git add Llms/prompt_engineering_guide.py Llms/Prompt_Engineering_Guide.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add prompt engineering guide module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch9 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring and keep behavior minimal.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch9.md
git commit -m "feat: add redpajama starcoder and prompt engineering guide modules"
```
