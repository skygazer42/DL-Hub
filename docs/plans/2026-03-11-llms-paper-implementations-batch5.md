# LLM Paper Implementations Batch 5 Plan

**Goal:** Extend `Llms/` with paper-shaped implementations for HELM, The Pile, and The Stack, covering holistic multi-metric evaluation, corpus mixture construction, and permissive-license code data curation.

**Architecture:** Implement `HELM` as a benchmark/runtime abstraction rather than a model backbone, exposing scenarios, metrics, reports, and prompt/completion logging. Implement `Pile` as a dataset-mixture abstraction with the paper’s 22 constituent sources and normalized sampling logic. Implement `The Stack` as a code-corpus pipeline with permissive-license filtering, near-deduplication, language accounting, and opt-out removal.

**Tech Stack:** Python 3.10, PyTorch-compatible Python utilities, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for HELM, Pile, and The Stack

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.helm`, `Llms.pile`, and `Llms.the_stack` exist
- title-case aliases `Llms.HELM`, `Llms.Pile`, and `Llms.The_Stack` exist
- `HELM` exposes scenario/metric/report abstractions and multi-metric coverage
- `Pile` exposes 22 component sources with normalized shares
- `The Stack` exposes permissive-license filtering, near-deduplication, and opt-out removal

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
git commit -m "test: add batch5 llms paper coverage"
```

### Task 2: Implement HELM evaluation abstractions

**Files:**
- Create: `Llms/helm.py`
- Create: `Llms/HELM.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- metric categories include the paper’s 7 general metrics
- evaluator computes scenario-metric coverage
- report preserves raw prompts and completions for transparency

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on HELM-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `HELMScenario`
- `HELMRun`
- `HELMEvaluator`
- `HELMReport`

Keep scope tight:
- focus on scenario/metric taxonomy and report standardization
- do not implement a full external benchmark runner

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: HELM tests PASS.

**Step 5: Commit**

```bash
git add Llms/helm.py Llms/HELM.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add helm paper module"
```

### Task 3: Implement The Pile corpus mixture

**Files:**
- Create: `Llms/pile.py`
- Create: `Llms/Pile.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- default registry contains 22 components
- component shares normalize to 1.0
- large sources like `Pile-CC` outrank small ones like `Enron Emails`

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Pile-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `PileComponent`
- `PileConfig`
- `PileMixture`
- `canonical_pile_components`

Scope:
- encode paper component metadata and normalized weights
- provide a deterministic sampler/count allocator

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Pile tests PASS.

**Step 5: Commit**

```bash
git add Llms/pile.py Llms/Pile.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add pile paper module"
```

### Task 4: Implement The Stack code-corpus pipeline

**Files:**
- Create: `Llms/the_stack.py`
- Create: `Llms/The_Stack.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- non-permissive licenses are filtered out
- near-duplicate files collapse under a configurable threshold
- developer opt-out removes repository files
- language volume summary is exposed

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on The Stack-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `TheStackConfig`
- `StackFile`
- `TheStackDataset`
- `NearDeduplicator`

Scope:
- focus on permissive-license filtering, near-deduplication, and opt-out governance
- do not implement remote crawling or GitHub ingestion

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: The Stack tests PASS.

**Step 5: Commit**

```bash
git add Llms/the_stack.py Llms/The_Stack.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add the-stack paper module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch5 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch5.md
git commit -m "feat: add helm pile and the-stack paper modules"
```
