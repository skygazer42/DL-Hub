# Lesson 31: Compact Diffusion Reference Editing

This lesson demonstrates a compact diffusion editing setup with three aligned tensors:

- `source`: the original synthetic image whose layout should be preserved.
- `reference`: a separate appearance image that supplies texture cues.
- `target`: the edited image that keeps source structure while borrowing reference appearance.

The model predicts diffusion noise on noised `target`, conditioned on `source` and `reference`.

Run locally with:

```bash
python -m tracks.generative.lesson_31_compact_diffusion_reference_editing.train --epochs 1 --device cpu
```

Artifacts are written to:

`outputs/generative/lesson_31_compact_diffusion_reference_editing/<run_name>/`

including:

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `reference_edit_triplets.pt`
- `checkpoints/checkpoint.pt`
