# Lesson 50: Compact Person-Pose VLM Reasoning

This lesson regresses a six-value pose description from a rendered person image and a fixed pose
query. The target contains lean, left/right arm, torso, and left/right leg factors normalized to
`[-1, 1]`; the stick-figure renderer deterministically turns those factors into grayscale images.

## Implementation

- `data.py` renders joints and limbs, tokenizes the pose query, and creates a seeded split.
- `model.py` fuses a two-layer CNN with a masked text embedding and predicts six values through a
  `tanh` regression head.
- `train.py` uses Smooth L1 plus a weighted pose MAE and logs mean MAE for train and evaluation.

## Quick Run

```bash
python -m tracks.multimodal.lesson_50_person_pose_vlm_reasoning.train \
  --epochs 1 --num-samples 64 --batch-size 8 --image-size 64 \
  --max-train-batches 2 --max-eval-batches 1 --device cpu --run-name smoke
```

## Outputs and Acceptance

The run directory is `outputs/multimodal/lesson_50_person_pose_vlm_reasoning/<run_name>/`. A
successful run writes `config.json`, `vocab.json`, `metrics.jsonl`, `logs/train.log`, and
`checkpoints/checkpoint.pt`. Acceptance requires finite loss plus non-negative `train_mean_mae` and
`eval_mean_mae` values.
