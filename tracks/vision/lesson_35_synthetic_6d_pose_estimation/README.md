# Lesson 35: Synthetic 6D Pose Estimation

This lesson uses simple rendered object silhouettes to regress a pose vector composed of a 6D
rotation representation and a 3D translation. The target is deliberately compact, but it still
captures the standard structure used by modern 6D pose pipelines.

## What It Teaches

- synthetic image generation for pose supervision
- compact CNN-based pose regression
- evaluation of normalized pose-vector error in a compact setup
