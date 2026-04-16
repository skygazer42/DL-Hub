# Lesson 36: Toy Diffusion Layout-Subject Fusion

This lesson models a compact layout-preserving generation setup with separate subject-style conditioning.
Each synthetic sample provides:

- `layout`: a structural guide that defines the geometry to preserve.
- `subject_style`: a separate style signal carrying appearance and texture cues.
- `target`: the fused image that follows `layout` while borrowing the subject-style appearance.

The model predicts diffusion noise on noised `target`, conditioned on `layout` and `subject_style`.

## Run

```bash
python -m tracks.generative.lesson_36_toy_diffusion_layout_subject_fusion.train --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_36_toy_diffusion_layout_subject_fusion/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `layout_subject_fusion_triplets.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
