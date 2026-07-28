# Lesson 35: Compact Diffusion Box-Mask Editing

This lesson models a small diffusion editing task conditioned on a source image and a rectangular
box mask. The target preserves the source outside the box while rewriting the boxed region with a
deterministic synthetic transform.

The batch contract is deliberately small:

- `source`: grayscale conditioning image
- `box_mask`: binary spatial edit region
- `target`: edited reconstruction target

Run:

```bash
python -m tracks.generative.lesson_35_compact_diffusion_box_mask_editing.train --epochs 1 --device cpu
```

Outputs are written to
`outputs/generative/lesson_35_compact_diffusion_box_mask_editing/<run_name>/`.
