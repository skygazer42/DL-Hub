# Lesson 44: Compact Diffusion Layout-Lighting Fusion

This lesson defines a compact conditional diffusion task that synthesizes a small RGB image from a
grayscale layout map plus a compact lighting code. The layout controls scene structure, while the
lighting code controls light direction, ambient fill, and warm-versus-cool illumination.

Batch contract:

- `layout`: `(B, 1, H, W)` float tensor in `[0, 1]`
- `lighting_code`: `(B, 4)` float vector for light direction and illumination style
- `target`: `(B, 3, H, W)` float tensor in `[0, 1]`

Run:

```bash
python -m tracks.generative.lesson_44_compact_diffusion_layout_lighting_fusion.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/generative/lesson_44_compact_diffusion_layout_lighting_fusion/<run_name>/`.
