# Lesson 43: Compact Diffusion Path-Mask Editing

This lesson trains a tiny diffusion model that edits only the region indicated by a synthetic path
mask. The dataset returns `(source, path_mask, target)` triplets, and training exports both final
samples and a `path_mask_triplets.pt` artifact for inspection.

## Run

```bash
python -m tracks.generative.lesson_43_compact_diffusion_path_mask_editing.train --epochs 1 --device cpu
```

Outputs are written to `outputs/generative/lesson_43_compact_diffusion_path_mask_editing/<run_name>/`.
