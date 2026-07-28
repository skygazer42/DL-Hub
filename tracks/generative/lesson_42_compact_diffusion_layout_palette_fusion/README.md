# Lesson 42: Compact Diffusion Layout-Palette Fusion

This lesson models a compact conditional diffusion task that synthesizes a small RGB image from a
grayscale layout map plus a palette code. The layout controls structure, while the palette code
controls the foreground and background colors blended into the generated target.

Batch contract:

- `layout`: `(B, 1, H, W)` float tensor in `[0, 1]`
- `palette_code`: `(B, 6)` float vector with foreground/background RGB values
- `target`: `(B, 3, H, W)` float tensor in `[0, 1]`

Run:

```bash
python -m tracks.generative.lesson_42_compact_diffusion_layout_palette_fusion.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/generative/lesson_42_compact_diffusion_layout_palette_fusion/<run_name>/`.
