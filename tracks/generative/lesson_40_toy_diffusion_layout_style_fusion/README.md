# Lesson 40: Toy Diffusion Layout-Style Fusion

This lesson models a small diffusion generation task conditioned on a spatial layout image plus a
compact style code vector. The layout provides coarse structure, while the style code controls the
synthetic appearance pattern injected into the structure.

Batch contract:

- `layout`: `(B, 1, H, W)` float tensor in `[0, 1]`
- `style_code`: `(B, 3)` float vector encoding a small style mixture
- `target`: `(B, 1, H, W)` float tensor in `[0, 1]`

Run:

```bash
python -m tracks.generative.lesson_40_toy_diffusion_layout_style_fusion.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/generative/lesson_40_toy_diffusion_layout_style_fusion/<run_name>/`.
