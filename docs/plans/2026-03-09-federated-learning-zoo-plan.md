# Federated Learning Zoo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a first federated-learning algorithm zoo with classic optimization families, timeline metadata, CLI discovery, and smoke tests.

**Architecture:** Create a new top-level `dlhub/federated/` package that exposes one family per file, plus a lightweight simulation-oriented strategy interface for one communication round. Mirror the existing zoo pattern used elsewhere in the repo: timeline metadata, AST-discovered builders, `scripts/*_zoo.py` CLI, and pytest smoke coverage.

**Tech Stack:** Python 3.10+, PyTorch, pytest, existing `dlhub` repository conventions.

---

### Task 1: Add the failing federated-zoo test surface

**Files:**
- Create: `tests/test_dlhub_federated_zoo.py`
- Create: `tests/test_dlhub_federated_algorithms.py`
- Create: `tests/test_dlhub_federated_timeline.py`

**Step 1: Write the failing test**

- Assert a new zoo lists representative ids for `fedavg`, `fedprox`, `scaffold`, `fednova`, `moon`, `pfedme`.
- Assert the timeline covers those families with year/group metadata.
- Assert each representative builder can run a single simulated round and returns finite tensors.

**Step 2: Run test to verify it fails**

Run:

```bash
pytest -q tests/test_dlhub_federated_zoo.py tests/test_dlhub_federated_algorithms.py tests/test_dlhub_federated_timeline.py
```

Expected:
- import failures for missing federated package/zoo/timeline

**Step 3: Write minimal implementation**

- Do not add lesson code.
- Only add enough structure so failures point at missing builders/metadata.

**Step 4: Run test to verify the failure surface is precise**

Run the same pytest command again.

**Step 5: Commit**

```bash
git commit -m "test: add federated learning zoo expectations"
```

### Task 2: Add the federated package, timeline, and CLI

**Files:**
- Create: `dlhub/federated/__init__.py`
- Create: `dlhub/federated/_common.py`
- Create: `dlhub/federated/_timeline.py`
- Create: `dlhub/federated_zoo.py`
- Create: `scripts/federated_zoo.py`

**Step 1: Write the failing test**

- Reuse the red tests from Task 1.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_federated_timeline.py tests/test_dlhub_federated_zoo.py
```

**Step 3: Write minimal implementation**

- Add a `FederatedBuildConfig`.
- Add a simulation-friendly base strategy interface and deterministic round runner.
- Add timeline metadata for the first batch.
- Add CLI support for `--list`, `--timeline`, and `--smoke`.

**Step 4: Run test to verify GREEN**

Run the same pytest command and confirm it passes.

**Step 5: Commit**

```bash
git commit -m "feat: add federated learning zoo core"
```

### Task 3: Implement the first federated families

**Files:**
- Create: `dlhub/federated/fedavg.py`
- Create: `dlhub/federated/fedprox.py`
- Create: `dlhub/federated/scaffold.py`
- Create: `dlhub/federated/fednova.py`
- Create: `dlhub/federated/moon.py`
- Create: `dlhub/federated/pfedme.py`

**Step 1: Write the failing test**

- Use the representative algorithm smoke tests from Task 1.

**Step 2: Run test to verify RED**

Run:

```bash
pytest -q tests/test_dlhub_federated_algorithms.py
```

**Step 3: Write minimal implementation**

- Keep implementations toy-first and simulation-first.
- Each file must expose `_VARIANTS`, `build_*_strategy(...)`, and a `__main__` smoke path.
- Use shared helpers instead of duplicating round-generation logic.

**Step 4: Run test to verify GREEN**

Run:

```bash
pytest -q tests/test_dlhub_federated_zoo.py tests/test_dlhub_federated_algorithms.py tests/test_dlhub_federated_timeline.py
```

**Step 5: Commit**

```bash
git commit -m "feat: add first federated learning strategy batch"
```
