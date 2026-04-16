# Lesson 82: Synthetic Layout Generation

This lesson introduces a tiny, CPU-friendly layout generation pipeline:

- generate synthetic condition maps from sparse object hints
- predict dense RGB layout maps plus occupancy masks
- train with a minimal backbone wired from `dlhub.vision.layout_generation`

The setup is intentionally toy-first and smoke-testable so it can serve as a direct
teaching bridge from layout families in `dlhub/vision/layout_generation` to
track-level training/evaluation loops.
