# Lesson 08: Perceiver-Resampler-Lite Compact VLM

This lesson introduces fixed-latent resampling over many visual tokens:

- build one full-scene view plus four quadrant crops
- encode all views into one large visual token pool
- resample that pool into a fixed set of latent tokens
- answer a short question about the scene

## Run

```bash
python -m tracks.multimodal.lesson_08_perceiver_resampler_compact_vlm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_08_perceiver_resampler_compact_vlm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Increase `num_latents` and compare exact match.
2. Remove the crop views and measure the drop from full-scene-only input.
3. Compare this lesson directly against lesson 7 on the same QA family.
