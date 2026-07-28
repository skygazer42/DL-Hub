# Lesson 33: Compact Diffusion Masked Reference Editing

This lesson extends the earlier reference-editing setup with an explicit spatial mask. Each
synthetic training example contains a `source` image, a `reference` texture image, a binary `mask`,
and a `target` image that preserves the source outside the mask while importing reference style
inside the masked region.

The diffusion model predicts noise for the target while conditioning on all four signals:

- `xt`: noised target image
- `source`: source layout image
- `reference`: reference appearance image
- `mask`: spatial edit region

## Run

```bash
python -m tracks.generative.lesson_33_compact_diffusion_masked_reference_editing.train --epochs 1 --device cpu
```

Outputs land under
`outputs/generative/lesson_33_compact_diffusion_masked_reference_editing/<run_name>/`.
