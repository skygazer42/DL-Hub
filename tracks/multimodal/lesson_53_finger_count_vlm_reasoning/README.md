# Lesson 53: Finger Count VLM Reasoning (Compact)

This lesson is a small multimodal classification exercise:

- Input: a synthetic grayscale hand-like image + a short query prompt
- Output: a finger-count class in `0..5` (6-way classification)

It follows the same minimal pattern as lessons 51-52:

- `data.py`: synthetic dataset + tiny vocab + dataloaders
- `model.py`: tiny vision+text encoders with a classifier head
- `train.py`: training loop that writes `config.json`, `vocab.json`, `metrics.jsonl`, and a checkpoint

