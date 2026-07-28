# Lesson 51: Hand Pose VLM Reasoning (Compact)

This lesson extends the multimodal reasoning arc to a hand-centric pose target. A synthetic
grayscale hand skeleton is rendered deterministically and paired with a short query. The model
regresses ten normalized (x, y) keypoints (wrist + finger base/tip anchors).

Run:

```bash
python -m tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.train --device cpu --epochs 1
```

Outputs are written to:

`outputs/multimodal/lesson_51_hand_pose_vlm_reasoning/<run_name>/`

