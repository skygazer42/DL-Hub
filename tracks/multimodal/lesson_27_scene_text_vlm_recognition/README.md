# Lesson 27: Compact Scene-Text VLM Recognition

This lesson provides a compact, CPU-friendly setup for recognizing short words from
synthetic scene images with a tiny vision-language recognizer.

- generate synthetic image patches that encode one scene word
- condition recognition with a short text prompt (`read the scene text`)
- fuse visual and text features
- classify one of four words: `alpha`, `beta`, `gamma`, `delta`

## Run

```bash
python -m tracks.multimodal.lesson_27_scene_text_vlm_recognition.train \
  --epochs 1 \
  --num-samples 64 \
  --batch-size 8 \
  --image-size 24 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```

## Outputs

`outputs/multimodal/lesson_27_scene_text_vlm_recognition/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`
