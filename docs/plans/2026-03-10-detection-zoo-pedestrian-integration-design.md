# Detection Zoo Pedestrian Presets — Script Integration Design

## Summary

Integrate the pedestrian detection presets (`dldet:pedestrian_*`) into the developer-facing
CLI workflow by extending `scripts/detection_zoo.py` so contributors can quickly verify:

- all pedestrian preset arch ids are discoverable (`--list --search pedestrian`)
- each preset can run a forward pass
- (optionally) each preset supports backward (`--backward`) by building a differentiable scalar loss

This provides a fast, repo-local integration check without relying on pytest.

## Goals

- Add a **single command** that smokes all presets:
  - `python scripts/detection_zoo.py --smoke-all --search pedestrian`
  - `python scripts/detection_zoo.py --smoke-all --search pedestrian --backward`
- Keep smoke runtime small (CPU-friendly defaults).
- Allow CI coverage via a lightweight subprocess-based pytest.

## Non-Goals

- No decoding/NMS or mAP/AP evaluation in the script.
- No training logic (handled by lessons).
- No attempts to unify output schemas across all detectors (they intentionally differ).

## CLI Design

Extend `scripts/detection_zoo.py` with:

- `--smoke-all`: run smoke for every listed (and optionally filtered) arch id.
- `--backward`: run `loss.backward()` in addition to forward, to validate gradient flow.
- `--keep-going`: when used with `--smoke-all`, continue after failures and return non-zero if any failed.

Behavior:

- Listing: unchanged.
- Single smoke (`--smoke <arch>`): unchanged by default; `--backward` adds backward check.
- Multi smoke (`--smoke-all`):
  - iterate through `arches` (already filtered by `--search`)
  - print per-arch status
  - if `--keep-going` is absent, fail-fast on first error
  - if `--keep-going` is present, collect failures and summarize at the end

## Backward Loss

Define a minimal differentiable scalar:

- For tensor outputs: mean of float32 cast
- For dict/list/tuple outputs: sum of scalar means recursively

This matches the repo’s existing “compact smoke” convention.

## Testing

Add `tests/test_scripts_detection_zoo_pedestrian.py`:

- `--list --search pedestrian` returns code 0 and lists known preset ids.
- `--smoke-all --search pedestrian --backward` returns code 0 (skipped if torch missing).

