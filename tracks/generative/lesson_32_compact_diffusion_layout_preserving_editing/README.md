# Lesson 32: Compact Diffusion Layout-Preserving Editing

This lesson demonstrates a compact conditional diffusion setup for layout-preserving editing.
Each synthetic sample provides:

- `layout`: a structural guide image that should be preserved.
- `edit`: an edit condition map describing where and how strongly local changes are applied.
- `target`: the edited image that follows `edit` while preserving global layout structure.

The model predicts diffusion noise on noised `target`, conditioned on `layout` and `edit`.

## Run

```bash
python -m tracks.generative.lesson_32_compact_diffusion_layout_preserving_editing.train --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_32_compact_diffusion_layout_preserving_editing/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `layout_editing_triplets.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
