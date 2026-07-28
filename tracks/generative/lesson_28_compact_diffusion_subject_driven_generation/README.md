# Lesson 28: Compact Diffusion Subject-Driven Generation

This lesson demonstrates a compact conditional diffusion setup for subject-driven generation.
Each synthetic sample provides:

- `subject`: an identity reference image carrying subject appearance.
- `guidance`: a layout-like map indicating where the subject should appear.
- `target`: the composed image matching the guidance while preserving subject texture cues.

The model predicts diffusion noise on noised `target`, conditioned on `subject` and `guidance`.

## Run

```bash
python -m tracks.generative.lesson_28_compact_diffusion_subject_driven_generation.train --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_28_compact_diffusion_subject_driven_generation/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `conditioning.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
