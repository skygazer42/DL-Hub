# Vision Backbones “100 Algorithms” Design

> Date: 2026-03-03  
> Scope: `dlhub/vision/backbones/`

## Goal

The current `dlhub/vision/backbones/` directory contains only a handful of large “kitchen sink” modules
(`cnn.py`, `extra_cnn.py`, `transformers.py`, …). The user requirement is to expand this directory with
~100 **algorithm-family** backbone implementations, where:

- **One algorithm family = one `.py` file**
- **Variants live in the same file** (e.g. `resnet18/resnet34/resnet50` inside `resnet.py`)
- Each file contains a real `torch.nn.Module` implementation (no `torchvision`/`timm` model imports)
- Each file provides an `if __name__ == "__main__"` random-forward smoke test

This design focuses on adding many self-contained, readable backbones without breaking the existing
local vision zoo (`dlhub/vision/local_zoo.py`) or tests.

## Non-goals (for this iteration)

- Rewriting the existing local zoo registry to automatically discover all new backbones
- Removing or heavily refactoring the existing large backbone modules (API stability first)
- Adding pretrained weights / downloads

## File layout

Add a small shared block library used by new backbones:

- `dlhub/vision/backbones/_blocks.py`
  - Tiny helpers that are safe to share (conv+bn+act, SE/ECA/CBAM-like attention blocks, etc.)
  - Only depends on `torch` / `torch.nn`

Then add many algorithm-family modules directly under:

- `dlhub/vision/backbones/<algorithm_name>.py`

Each algorithm module follows this convention:

- Defines one primary network class, e.g. `class ResNet(nn.Module): ...`
- Exposes a `build_<algorithm>_classifier(...) -> nn.Module` helper (optional but recommended)
- Exposes a `_VARIANTS` dict (or equivalent) that maps variant names to specs
- Includes a simple `__main__` smoke:
  - Builds one or more variants
  - Runs `torch.randn(2, in_channels, 64, 64)` through the model
  - Prints output shapes

## Compatibility

Existing production code uses:

- `dlhub/vision/backbones/__init__.py` to import the current set of `build_*_classifier` functions
- `dlhub/vision/local_zoo.py` to register arch ids for the “dl:” local zoo

To keep the repo stable:

- New modules are added without changing existing imports by default.
- Optional follow-up: selectively export and register new families once validated.

## Testing strategy

1. Keep existing tests passing:
   - `pytest -q tests/test_dlhub_vision_local_zoo.py`
2. Add a lightweight import/compile check for new modules (optional):
   - `python -m py_compile dlhub/vision/backbones/*.py`
3. Each new algorithm file has its own `__main__` smoke that can be run directly.

