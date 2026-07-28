# Lesson 85: Synthetic Scene Text Spotting

This lesson provides a compact-first, CPU-friendly scene text spotting loop:

- synthesize one text region per image with a dense spotting score map target
- decode a short fixed-length token sequence for recognition
- reuse a local `dlhub.vision.scene_text_spotting` family as the spotting backbone
- optimize a compact multi-task objective (detection + recognition + auxiliary token)

Run:

```bash
python -m tracks.vision.lesson_85_synthetic_scene_text_spotting.train \
  --epochs 1 \
  --max-train-batches 2 \
  --max-eval-batches 1 \
  --device cpu
```
