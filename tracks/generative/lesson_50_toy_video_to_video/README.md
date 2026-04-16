# Lesson 50: Toy Video-to-Video

This lesson introduces a minimal video-to-video pipeline:

- generate paired synthetic source/target RGB clips `(C, T, H, W)`
- train a tiny 3D-convolutional translation model from `dlhub.generative.video_to_video`
- optimize reconstruction with lightweight regularization on residual motion

The setup is intentionally toy-first and CPU-friendly, so it can be used as a smoke-testable
teaching scaffold for video translation workflows.
