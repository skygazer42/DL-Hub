# Lesson 38: Synthetic Face Detection

This lesson introduces face detection as normalized bounding-box regression over synthetic face
renderings. Each sample includes one rendered face and a target box in `[x_min, y_min, x_max,
y_max]` format.

## What It Teaches

- synthetic box label generation for single-face detection
- compact convolutional detector for normalized box regression
- pixel-space L1 box error tracking during evaluation
