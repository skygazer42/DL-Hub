# LLM Paper Implementations Batch 11 Plan

**Goal:** Extend `Llms/` with final implementation-friendly abstractions for Bard, AI Bubbles, and OASST1, covering an early collaborative generative-AI product interface, a model-size registry, and an alignment-dataset alias for OpenAssistant.

**Architecture:** Implement `Bard` as a product-facing collaboration wrapper that models the early experiment framing, user collaboration modes, and principle-aware responses rather than a new neural backbone. Implement `AI Bubbles` as a registry of model-scale entries with open/closed filtering and Chinchilla-scale flags derived from the local bubble chart. Implement `OASST1` as a dataset alias around `OpenAssistant` that exposes the naming and alignment-data identity of the OpenAssistant Conversations Dataset.

**Tech Stack:** Python 3.10, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for Bard, AI Bubbles, and OASST1

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.bard`, `Llms.ai_bubbles`, and `Llms.oasst1` exist
- title-case aliases `Llms.Bard`, `Llms.AI_Bubbles`, and `Llms.OASST1` exist
- `Bard` exposes collaboration modes and early-experiment metadata
- `AI Bubbles` exposes model-size entries with availability/open/closed filters
- `OASST1` exposes OpenAssistant-style conversation dataset metadata under the OASST1 name

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
git commit -m "test: add batch11 llms paper coverage"
```

### Task 2: Implement Bard collaboration wrapper

**Files:**
- Create: `Llms/bard.py`
- Create: `Llms/Bard.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- model metadata marks Bard as an early experiment
- collaboration modes include productivity, creativity, and curiosity
- response formatting keeps principle-aware structure

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Bard-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `BardConfig`
- `BardMode`
- `BardSession`
- `format_bard_response`

Keep scope tight:
- model the local overview PDF’s product framing and collaboration categories
- avoid inventing unsupported backend architecture details

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Bard tests PASS.

**Step 5: Commit**

```bash
git add Llms/bard.py Llms/Bard.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add bard overview module"
```

### Task 3: Implement AI Bubbles model-scale registry

**Files:**
- Create: `Llms/ai_bubbles.py`
- Create: `Llms/AI_Bubbles.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- canonical entries include Bard, GPT-4, LLaMA, and PaLM
- registry can filter open vs closed models
- registry can select Chinchilla-scale entries

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on AI-Bubbles-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `AIBubbleEntry`
- `AIBubblesRegistry`
- `canonical_ai_bubbles_entries`

Scope:
- encode the local bubble chart as queryable metadata
- keep it as a registry rather than a visual plotting system

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: AI Bubbles tests PASS.

**Step 5: Commit**

```bash
git add Llms/ai_bubbles.py Llms/AI_Bubbles.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add ai bubbles registry module"
```

### Task 4: Implement OASST1 dataset alias

**Files:**
- Create: `Llms/oasst1.py`
- Create: `Llms/OASST1.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- OASST1 dataset metadata identifies the OpenAssistant Conversations Dataset
- alias wraps conversation trees and preference pairs in the same style as `OpenAssistant`
- exported API name reflects OASST1 rather than the generic OpenAssistant name

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on OASST1-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `OASST1Config`
- `OASST1Dataset`

Scope:
- reuse `OpenAssistant` abstractions where appropriate
- keep the implementation minimal and name-aligned with the dataset note

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: OASST1 tests PASS.

**Step 5: Commit**

```bash
git add Llms/oasst1.py Llms/OASST1.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add oasst1 dataset module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch11 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring and keep behavior minimal.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch11.md
git commit -m "feat: add bard ai bubbles and oasst1 modules"
```
