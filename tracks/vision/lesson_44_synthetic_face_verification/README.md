# Lesson 44: Synthetic Face Verification

This lesson trains a compact pairwise verifier that predicts whether two rendered face crops
belong to the same synthetic identity. Each sample contains `(image_a, image_b, label)` where
`label=1` means same identity and `label=0` means different identities.

The lesson covers:
- deterministic paired synthetic face generation
- a small shared-backbone verifier with 2-class logits
- config, metrics, logs, and checkpoint outputs

Run locally with:

```bash
python -m tracks.vision.lesson_44_synthetic_face_verification.train --device cpu --epochs 1
```
