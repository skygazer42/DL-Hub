# Lesson 34: Toy Diffusion Layout-Reference Fusion

This lesson extends the recent editing block with a compact layout-reference fusion setup.
Each synthetic sample provides:

- `layout`: a structural guide that defines the geometry to preserve.
- `reference`: an appearance image that contributes texture and tone cues.
- `target`: the fused image that follows `layout` while borrowing reference appearance.

The model predicts diffusion noise on noised `target`, conditioned on `layout` and `reference`.

## Run

```bash
python -m tracks.generative.lesson_34_toy_diffusion_layout_reference_fusion.train --epochs 1 --device cpu
```

## Outputs

`outputs/generative/lesson_34_toy_diffusion_layout_reference_fusion/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `samples.pt`
- `layout_reference_fusion_triplets.pt`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
