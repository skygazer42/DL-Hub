# Lesson 29: Compact Human-Object Interaction Reasoning

This lesson introduces a compact multimodal reasoning setup over synthetic
region features and a text query describing a human-object relation.

- synthesize region features and boxes for a person, objects, and distractors
- encode a short query such as `person holding cup`
- fuse pooled region interaction state with text state
- classify binary answer: relation true (`yes`) or false (`no`)

The task is intentionally compact-first and CPU-friendly.

## Run

```bash
python -m tracks.multimodal.lesson_29_human_object_interaction_reasoning.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --num-regions 6 \
  --feature-dim 16 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```
