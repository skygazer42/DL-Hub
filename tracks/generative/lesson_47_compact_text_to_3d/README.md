# Lesson 47: Compact Text-to-3D

This lesson builds a tiny, CPU-friendly text-to-3D pipeline:

- synthetic 32-dim text feature vectors
- compact density-grid and mesh-token targets
- a wrapper over `dlhub.generative.text_to_3d` builders
- single-file training loop with standard DL-Hub outputs

Run:

```bash
python -m tracks.generative.lesson_47_compact_text_to_3d.train --epochs 1 --device cpu
```
