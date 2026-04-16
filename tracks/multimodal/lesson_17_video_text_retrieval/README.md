# Lesson 17: Video-Text Retrieval Toy CLIP

This lesson follows the temporal grounding block with a lightweight video-text retrieval task:

- render short synthetic videos with one colored shape following a motion pattern
- encode each frame with a tiny CNN and mean-pool over time
- encode short text captions or query variants with a token embedding encoder
- train with a symmetric contrastive retrieval loss and report retrieval metrics

## Run

```bash
python -m tracks.multimodal.lesson_17_video_text_retrieval.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_17_video_text_retrieval/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
