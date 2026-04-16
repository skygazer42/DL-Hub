# Lesson 37: Toy Diffusion Polygon-Mask Editing

This lesson trains a tiny conditional diffusion model that edits only the polygon-selected region
of a synthetic grayscale image. The setup mirrors mask-guided image editing while remaining light
enough for CPU smoke tests.

Run:

```bash
python -m tracks.generative.lesson_37_toy_diffusion_polygon_mask_editing.train --epochs 1 --device cpu
```

Outputs are written to
`outputs/generative/lesson_37_toy_diffusion_polygon_mask_editing/<run_name>/`.
