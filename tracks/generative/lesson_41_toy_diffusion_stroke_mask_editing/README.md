# Lesson 41: Toy Diffusion Stroke-Mask Editing

This lesson trains a tiny diffusion model that edits only the region indicated by a synthetic stroke
mask. The dataset returns `(source, stroke_mask, target)` triplets, and training exports both the
final samples and a `stroke_mask_triplets.pt` artifact for inspection.

## Run

```bash
python -m tracks.generative.lesson_41_toy_diffusion_stroke_mask_editing.train --epochs 1 --device cpu
```

Outputs are written to `outputs/generative/lesson_41_toy_diffusion_stroke_mask_editing/<run_name>/`.
