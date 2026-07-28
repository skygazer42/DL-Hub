# LLM Paper Implementations Batch 4 Plan

**Goal:** Extend `Llms/` with paper-shaped implementations for ZeRO, Parameter Server, and FED, covering optimizer-state partitioning, distributed parameter serving, and fast multi-query decoding.

**Architecture:** Represent `ZeRO` as a training-state partitioning abstraction rather than another language backbone. Represent `Parameter Server` as a distributed coordination/runtime module with consistency controls, sparse pulls, and worker push/pull APIs. Represent `FED` as a decoder-side multi-query attention module focused on incremental inference and KV-cache efficiency, not as a full duplicate decoder stack.

**Tech Stack:** Python 3.10, PyTorch, pytest, existing `Llms/` package conventions

---

### Task 1: Add failing tests for ZeRO, Parameter Server, and FED

**Files:**
- Modify: `tests/test_llms_papers.py`
- Modify: `tests/test_import_regressions.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that assert:
- `Llms.zero`, `Llms.parameter_server`, and `Llms.fed` exist
- title-case aliases `Llms.ZeRO`, `Llms.Parameter_Server`, and `Llms.FED` exist
- `ZeRO` exposes stage-wise optimizer/gradient/parameter partitioning behavior
- `Parameter Server` exposes sparse pull/push and bounded-staleness scheduling
- `FED` exposes multi-query attention with shared KV heads and incremental decoding

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
git commit -m "test: add batch4 llms paper coverage"
```

### Task 2: Implement ZeRO state partitioning

**Files:**
- Create: `Llms/zero.py`
- Create: `Llms/ZeRO.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add or refine tests to verify:
- stage 1 partitions optimizer state only
- stage 2 also partitions gradients
- stage 3 also partitions parameters
- all-gather style reconstruction can recover the original parameter vector

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on ZeRO-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `ZeROConfig`
- `ZeROPartitionPlan`
- `ZeROPartitioner`
- `ZeROEngine`

Keep scope tight:
- focus on model-state memory partitioning
- represent stage semantics directly
- expose enough helpers to test gather/reconstruction and per-rank ownership

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: ZeRO tests PASS.

**Step 5: Commit**

```bash
git add Llms/zero.py Llms/ZeRO.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add zero paper module"
```

### Task 3: Implement Parameter Server runtime

**Files:**
- Create: `Llms/parameter_server.py`
- Create: `Llms/Parameter_Server.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- worker API can pull sparse keys
- push updates mutate the shared server state
- staleness rules prevent workers from racing too far ahead under SSP-like control

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on Parameter Server-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `ParameterServerConfig`
- `ParameterServer`
- `ParameterWorker`
- `ConsistencyController`

Scope:
- model dense/sparse shared parameters as Python/Torch dictionaries or tensors
- expose asynchronous pull/push flavored API
- add a small logical-clock consistency mechanism rather than full networking

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: Parameter Server tests PASS.

**Step 5: Commit**

```bash
git add Llms/parameter_server.py Llms/Parameter_Server.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add parameter server module"
```

### Task 4: Implement FED multi-query incremental decoding

**Files:**
- Create: `Llms/fed.py`
- Create: `Llms/FED.py`
- Modify: `Llms/__init__.py`
- Modify: `tests/test_llms_papers.py`
- Test: `tests/test_llms_papers.py`

**Step 1: Write the failing test**

Add tests that verify:
- keys and values are shared across heads
- incremental decoding with cache matches full causal decoding
- cache footprint helper is smaller than standard multi-head attention

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FAIL on FED-specific assertions.

**Step 3: Write minimal implementation**

Implement:
- `FEDConfig`
- `FEDMultiQueryAttention`
- `FEDDecoderBlock`
- `FEDModel`

Scope:
- focus on the paper’s one-write-head/shared-KV idea
- keep the decoder small and faithful
- expose a cache-size estimator to make the decoding optimization testable

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py -q`
Expected: FED tests PASS.

**Step 5: Commit**

```bash
git add Llms/fed.py Llms/FED.py Llms/__init__.py tests/test_llms_papers.py
git commit -m "feat: add fed fast decoding module"
```

### Task 5: Final verification

**Files:**
- Test: `tests/test_llms_papers.py`
- Test: `tests/test_import_regressions.py`

**Step 1: Write the failing test**

No new tests beyond the import-regression additions from Task 1.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: FAIL until all batch4 modules are complete.

**Step 3: Write minimal implementation**

Finish any remaining import/export wiring.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_llms_papers.py tests/test_import_regressions.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add Llms tests/test_llms_papers.py tests/test_import_regressions.py docs/plans/2026-03-11-llms-paper-implementations-batch4.md
git commit -m "feat: add zero parameter-server and fed paper modules"
```
