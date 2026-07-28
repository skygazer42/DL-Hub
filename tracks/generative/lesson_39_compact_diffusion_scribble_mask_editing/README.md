# Lesson 39: Compact Diffusion Scribble-Mask Editing

This lesson trains a tiny conditional diffusion model that edits only the user-scribbled region of
an image. The synthetic dataset renders a source canvas, a sparse scribble mask, and a target image
whose edits are constrained to that scribbled area.

The implementation is intentionally lightweight so it can act as a smoke-testable teaching example
for scribble-guided image editing workflows.

## Run

```bash
python -m tracks.generative.lesson_39_compact_diffusion_scribble_mask_editing.train --epochs 1 --device cpu
```

Outputs land under `outputs/generative/lesson_39_compact_diffusion_scribble_mask_editing/<run_name>/`.
