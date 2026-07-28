# Lesson 25: Compact Vision-Language Navigation

This lesson introduces a compact vision-language navigation (VLN) setup in a tiny grid world.

- synthesize an observation with an agent cell and a goal cell
- encode a short instruction text with a direction token
- fuse visual and text features into a one-step navigation policy
- predict one of four actions: `north`, `south`, `east`, `west`

The setup is deterministic, CPU-friendly, and designed for fast local experiments.

## Run

```bash
python -m tracks.multimodal.lesson_25_vision_language_navigation.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --grid-size 7 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Outputs

`outputs/multimodal/lesson_25_vision_language_navigation/<run_name>/`

- `config.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
