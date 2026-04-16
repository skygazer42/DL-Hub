# Lesson 28: Document VLM Reasoning (Toy)

This lesson demonstrates a tiny document VLM that answers simple binary questions from
synthetic document text and layout-rendered image cues.

## What it covers

- Synthetic invoice-like documents with fields (`city`, `total`, `priority`)
- Text query answering into two classes (`low`, `high`)
- Minimal multimodal fusion of image, document tokens, and query tokens

## Run

```bash
python -m tracks.multimodal.lesson_28_document_vlm_reasoning.train --device cpu --epochs 1
```
