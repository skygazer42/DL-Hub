# Lesson 18: Synthetic Crowd Counting

This lesson builds a tiny CPU-friendly crowd-counting pipeline in pure PyTorch.
It generates synthetic grayscale scenes, supervises a per-pixel density map, and
reports image-level count metrics by summing the predicted density.
