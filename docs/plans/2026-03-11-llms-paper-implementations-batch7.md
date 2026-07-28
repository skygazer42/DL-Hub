# LLM Paper Implementations Batch 7 Plan

**Goal:** Extend `Llms/` with paper-shaped implementations for Chain-of-Thought prompting, GPT4All-J, and Multitask Prompted Training, covering reasoning-trace prompting, GPT-J-based assistant finetuning, and T0-style zero-shot task generalization.

**Architecture:** Implement `Chain-of-Thought` as a reasoning prompt and self-consistency abstraction rather than a backbone model, exposing rationale formatting, answer extraction, and majority-vote aggregation. Implement `GPT4All-J` as a GPT-J-backed assistant wrapper that captures the paper’s Apache-2 licensing, larger curated dataset, creative prompt augmentation, and CPU-friendly 4-bit deployment metadata. Implement `MTF` as a multitask prompted-training wrapper over a T5-style encoder-decoder model, modeling prompt templates, task-mixture sampling, and held-out task zero-shot evaluation without recreating the full T0 training stack.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for Chain-of-Thought, GPT4All-J, and MTF

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.chain_of_thought`, `Llms.gpt4all_j`, and `Llms.mtf` exist
- title-case aliases `Llms.Chain_of_Thought`, `Llms.GPT4All_J`, and `Llms.MTF` exist
- `Chain-of-Thought` exposes rationale-aware prompting, answer extraction, and self-consistency voting
- `GPT4All-J` exposes GPT-J backbone metadata, creative augmentation data sources, Apache-2 licensing, and 4-bit deployment metadata
- `MTF` exposes prompt templates, task-mixture construction, and a T0-style multitask wrapper

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
git commit -m "test: add batch7 llms paper coverage"
```

### Task 2: Implement Chain-of-Thought reasoning abstraction

**Files:**
- Create: `Llms/chain_of_thought.py`
- Create: `Llms/Chain_of_Thought.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- prompt builder includes demonstrations, question text, and a step-by-step cue
- answer extraction can recover the final answer from a rationale trace
- self-consistency voting prefers the majority final answer across sampled traces

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Chain-of-Thought-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `ChainOfThoughtExample`
- `ChainOfThoughtConfig`
- `format_chain_of_thought_prompt`
- `extract_final_answer`
- `SelfConsistencyDecoder`
- `ChainOfThoughtReasoner`

Keep scope tight:
- encode the paper’s intermediate-reasoning and sample-plus-vote mechanics
- avoid pretending this is a new pretrained backbone

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Chain-of-Thought tests PASS.

**Step 5: Commit**

```bash
git add Llms/chain_of_thought.py Llms/Chain_of_Thought.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add chain-of-thought paper module"
```

### Task 3: Implement GPT4All-J assistant wrapper

**Files:**
- Create: `Llms/gpt4all_j.py`
- Create: `Llms/GPT4All_J.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- prompt formatting supports assistant-style creative prompts
- metadata reflects GPT-J, Apache-2 licensing, creative augmentation, and the expanded curated dataset
- model wraps a `GPTJ`-like decoder or a repo-consistent GPT-style approximation with GPT-J metadata

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on GPT4All-J-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `GPT4AllJDataCuration`
- `GPT4AllJConfig`
- `format_gpt4all_j_prompt`
- `GPT4AllJModel`

Scope:
- model the paper’s GPT-J-based open assistant framing
- distinguish it cleanly from the earlier `GPT4All` LLaMA-distilled wrapper

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: GPT4All-J tests PASS.

**Step 5: Commit**

```bash
git add Llms/gpt4all_j.py Llms/GPT4All_J.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add gpt4all-j paper module"
```

### Task 4: Implement MTF multitask prompt-training wrapper

**Files:**
- Create: `Llms/mtf.py`
- Create: `Llms/MTF.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- prompt templates can materialize dataset examples into input and target text
- task mixtures can track seen vs held-out tasks for zero-shot evaluation
- model wraps a T5-style encoder-decoder and exposes multitask prompted-training metadata

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on MTF-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `PromptTemplate`
- `MTFTask`
- `MTFMixture`
- `MTFConfig`
- `MTFModel`

Scope:
- model the paper’s prompt-source and held-out-task methodology
- keep the wrapper compact and testable instead of reproducing the full benchmark release

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: MTF tests PASS.

**Step 5: Commit**

```bash
git add Llms/mtf.py Llms/MTF.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add mtf paper module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch7 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring and keep behavior minimal.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch7.md
git commit -m "feat: add chain-of-thought gpt4all-j and mtf paper modules"
```
