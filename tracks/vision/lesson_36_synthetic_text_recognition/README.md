# Lesson 36: Synthetic Text Recognition

This lesson implements OCR-style fixed-length text recognition from synthetic cropped word images.
It uses deterministic glyph templates with light noise and a compact CNN that predicts one token
per character slot.

## What It Teaches

- simple synthetic OCR data generation without external assets
- fixed-length sequence prediction from visual features
- exact-match sequence accuracy for text recognition tasks
