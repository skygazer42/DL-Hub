# LLM Paper Implementations Batch 10 Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `Llms/` with paper-shaped implementations for OpenAssistant, the LLM Survey, and the LLM Timeline, covering alignment conversation trees, systematized LLM taxonomy/evaluation, and milestone tracking for recent LLM releases.

**Architecture:** Implement `OpenAssistant` as an alignment-data abstraction with conversation trees, assistant turns, and preference ratings rather than a new backbone model. Implement `LLM Survey` as a taxonomy and benchmark registry that models the survey’s four major aspects, resource pointers, and major evaluation suites. Implement `LLM Timeline` as a compact milestone registry with date-based filtering over key model and dataset releases from the local timeline document.

**Tech Stack:** Python 3.10, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for OpenAssistant, LLM Survey, and LLM Timeline

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.openassistant`, `Llms.llm_survey`, and `Llms.llm_timeline` exist
- title-case aliases `Llms.OpenAssistant`, `Llms.LLM_Survey`, and `Llms.LLM_Timeline` exist
- `OpenAssistant` exposes conversation trees, quality ratings, and preference-pair extraction
- `LLM Survey` exposes the four major aspects, benchmark registry, and resource view
- `LLM Timeline` exposes dated milestones with year/month filtering

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
git commit -m "test: add batch10 llms paper coverage"
```

### Task 2: Implement OpenAssistant alignment-data abstraction

**Files:**
- Create: `Llms/openassistant.py`
- Create: `Llms/OpenAssistant.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- conversation trees can flatten messages in traversal order
- preference ratings can be converted into chosen/rejected pairs
- dataset metadata reflects multilingual volunteer alignment data

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on OpenAssistant-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `OpenAssistantMessage`
- `OpenAssistantPreference`
- `OpenAssistantConfig`
- `OpenAssistantConversationTree`
- `OpenAssistantDataset`

Keep scope tight:
- model the paper’s data and ranking structure
- avoid inventing a nonexistent new network architecture

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: OpenAssistant tests PASS.

**Step 5: Commit**

```bash
git add Llms/openassistant.py Llms/OpenAssistant.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add openassistant paper module"
```

### Task 3: Implement LLM Survey taxonomy and benchmark registry

**Files:**
- Create: `Llms/llm_survey.py`
- Create: `Llms/LLM_Survey.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- the four major aspects are pre-training, adaptation tuning, utilization, and capacity evaluation
- benchmark registry includes MMLU, BIG-bench, and HELM
- resource view exposes checkpoints, corpora, and tooling categories

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on LLM-Survey-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `SurveyAspect`
- `BenchmarkSuite`
- `SurveyResource`
- `LLMSurveyGuide`

Scope:
- encode the survey’s structural taxonomy and benchmark landscape
- keep it concise and metadata-driven

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: LLM Survey tests PASS.

**Step 5: Commit**

```bash
git add Llms/llm_survey.py Llms/LLM_Survey.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add llm survey paper module"
```

### Task 4: Implement LLM Timeline milestone registry

**Files:**
- Create: `Llms/llm_timeline.py`
- Create: `Llms/LLM_Timeline.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- canonical entries contain representative 2023 releases such as Bard, GPT4All-J, OpenAssistant, and StarCoder
- filtering by year and month returns the expected milestones
- timeline categories distinguish models, datasets, and surveys/tools

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on LLM-Timeline-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `TimelineEntry`
- `LLMTimeline`
- `canonical_llm_timeline_entries`

Scope:
- encode the local timeline file as a queryable milestone registry
- avoid over-expanding beyond the events named in the local document

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: LLM Timeline tests PASS.

**Step 5: Commit**

```bash
git add Llms/llm_timeline.py Llms/LLM_Timeline.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add llm timeline module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch10 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring and keep behavior minimal.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch10.md
git commit -m "feat: add openassistant survey and timeline paper modules"
```
