# Lesson 23: Compact Embodied Question Answering

This lesson introduces a compact embodied QA setup over short navigation trajectories.

- synthesize tiny trajectories and aligned egocentric observations
- ask a fixed navigation question: `where is the goal from final position`
- fuse observation, trajectory, and text features
- classify one of four answers: `left`, `right`, `up`, `down`

The task is intentionally compact-first, deterministic, and CPU-friendly.

## Run

```bash
python -m tracks.multimodal.lesson_23_embodied_question_answering.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Outputs

`outputs/multimodal/lesson_23_embodied_question_answering/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add obstacle tokens and ask whether the final step is collision-free.
2. Predict a two-token answer (`turn left`, `turn right`) with sequence decoding.
3. Replace the GRU aggregator with attention over trajectory steps.
