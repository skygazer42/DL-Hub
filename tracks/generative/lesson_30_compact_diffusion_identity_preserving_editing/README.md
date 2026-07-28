# Lesson 30: Compact Diffusion Identity-Preserving Editing

This lesson demonstrates a compact conditional diffusion setup for identity-preserving editing.
Each synthetic sample provides:

- `subject`: an identity reference image carrying stable appearance cues.
- `edit`: an edit condition map describing where and how strongly an edit should be applied.
- `target`: the edited image that follows `edit` while retaining subject identity texture.

The model predicts diffusion noise on noised `target`, conditioned on `subject` and `edit`.

## Run

```bash
python -m tracks.generative.lesson_30_compact_diffusion_identity_preserving_editing.train --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_30_compact_diffusion_identity_preserving_editing/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `editing_triplets.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
