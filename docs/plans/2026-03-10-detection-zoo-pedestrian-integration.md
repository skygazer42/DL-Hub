# Detection Zoo Pedestrian Presets Integration Plan

**Goal:** Extend `scripts/detection_zoo.py` so contributors can smoke all `dldet:pedestrian_*` presets (optionally with backward) via a single CLI command, and cover it with a small subprocess-based pytest.

**Architecture:** Add `--smoke-all` and `--backward` flags to `scripts/detection_zoo.py`. Implement a small recursive loss helper for backward checks. Add a pytest that runs the script with `--search pedestrian` to validate listing and smoke-all.

**Tech Stack:** Python 3.10+, PyTorch (`torch`), `pytest`, `subprocess`.

---

### Task 1: Write failing script test (subprocess)

**Files:**
- Create: `tests/test_scripts_detection_zoo_pedestrian.py`

**Step 1: Write a failing test**

```python
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_detection_zoo_lists_pedestrian_presets() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/detection_zoo.py", "--list", "--search", "pedestrian"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "dldet:pedestrian_fcos" in proc.stdout


def test_detection_zoo_smoke_all_pedestrian_presets_backward() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/detection_zoo.py",
            "--smoke-all",
            "--search",
            "pedestrian",
            "--backward",
            "--batch-size",
            "1",
            "--image-size",
            "64",
            "--num-classes",
            "1",
            "--width-mult",
            "0.5",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
```

**Step 2: Run the test to verify it fails**

Run: `pytest -q tests/test_scripts_detection_zoo_pedestrian.py`
Expected: FAIL because `--smoke-all` / `--backward` do not exist yet.

**Step 3: Commit**

```bash
git add tests/test_scripts_detection_zoo_pedestrian.py
git commit -m "test: add detection_zoo pedestrian smoke-all checks (failing)"
```

---

### Task 2: Add `--smoke-all` / `--backward` to `scripts/detection_zoo.py`

**Files:**
- Modify: `scripts/detection_zoo.py`

**Step 1: Update CLI args**

- Add `--smoke-all` (bool)
- Add `--backward` (bool)
- Add `--keep-going` (bool, only meaningful with `--smoke-all`)

**Step 2: Implement loss helper**

Add a private helper:

```python
def _sum_output_means(x) -> torch.Tensor:
    ...
```

Rules:
- tensor → float32 mean
- dict → sum over values
- list/tuple → sum over items

**Step 3: Implement smoke runner**

Add a helper `_run_smoke(arch_id: str) -> None` that:
- builds model via `build_local_model`
- runs forward
- if `--backward`: compute loss via `_sum_output_means` and `loss.backward()`

**Step 4: Implement `--smoke-all` loop**

- Iterate over `arches` (already filtered by `--search`)
- Fail-fast unless `--keep-going`
- Print a small summary and return non-zero if any failures

**Step 5: Run tests**

Run: `pytest -q tests/test_scripts_detection_zoo_pedestrian.py`
Expected: PASS

**Step 6: Commit**

```bash
git add scripts/detection_zoo.py
git commit -m "feat: add detection_zoo smoke-all and backward checks"
```

