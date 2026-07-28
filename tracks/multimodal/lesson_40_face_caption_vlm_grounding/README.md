# Lesson 40: Face Caption VLM Grounding

This lesson builds a compact multimodal grounding model that takes one synthetic face crop plus a
caption query and predicts whether the caption matches the face attributes.

## What It Teaches

- synthetic face-caption pair construction with controlled match/mismatch labels
- compact vision-language fusion for caption grounding
- binary training loop with metrics logging and checkpoint export
