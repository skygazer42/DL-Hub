# Lesson 03: LLaVA-Lite Visual Instruction VLM

This lesson upgrades the multimodal track to single-turn visual instruction following:

- encode an image into visual tokens
- project visual tokens into the language-model space
- condition a tiny decoder LM on the visual prefix plus instruction text
- generate a short answer

## Run

```bash
python -m tracks.multimodal.lesson_03_llava_compact_instruction_vlm.train --epochs 1 --max-train-batches 2 --max-eval-batches 1 --device cpu
```

## Outputs

`outputs/multimodal/lesson_03_llava_compact_instruction_vlm/<run_name>/`

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Exercises

1. Add new instruction types such as counting or conjunction questions.
2. Replace the recurrent decoder with a tiny Transformer decoder.
3. Add a separate visual projector depth setting and compare training stability.
