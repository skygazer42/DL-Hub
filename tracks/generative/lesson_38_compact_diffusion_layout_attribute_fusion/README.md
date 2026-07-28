# Lesson 38: Compact Diffusion Layout-Attribute Fusion

This lesson models a small diffusion generation task conditioned on a spatial layout image plus a
compact attribute vector. The layout provides coarse structure, while the attribute controls the
synthetic appearance pattern injected into the structure.

Batch contract:

- `layout`: `(B, 1, H, W)` float tensor in `[0, 1]`
- `attribute`: `(B, 4)` float vector encoding a small attribute mixture
- `target`: `(B, 1, H, W)` float tensor in `[0, 1]`

Run:

```bash
python -m tracks.generative.lesson_38_compact_diffusion_layout_attribute_fusion.train --epochs 1 --device cpu
```

Outputs are written under
`outputs/generative/lesson_38_compact_diffusion_layout_attribute_fusion/<run_name>/`.
