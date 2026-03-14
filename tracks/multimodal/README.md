# Multimodal Track

Goal: learn the core progression of modern vision-language modeling through small, runnable teaching implementations.

This track is intentionally ordered as a sixteen-step path:

1. align image and text in one embedding space
2. fuse image and text for generation and matching
3. condition a decoder-style language model on visual tokens and follow instructions
4. ground language to a spatial region in the image
5. predict a language-conditioned mask for the grounded region
6. answer a query from an interleaved few-shot image-text prompt
7. compress vision into learned query tokens before decoding
8. resample many visual tokens into a fixed latent budget
9. solve multiple tasks through one prompt-native text interface
10. detect text-specified categories even when the queried object may be absent
11. segment text-specified categories even when the queried object may be absent
12. read structured fields from a document image with prompt-conditioned text generation
13. answer prompt-conditioned questions over a short video by adding temporal aggregation
14. ground a text query to a temporal video segment with boundary prediction and proposal matching
15. ground a text query to a temporal video segment with dense 2D segment-map reasoning
16. ground a text query to a temporal video segment with fused multi-scale 2D segment-map reasoning

That progression maps directly to the sixteen current lessons:

- `lesson_01_clip_toy_retrieval/`
- `lesson_02_blip_toy_captioning/`
- `lesson_03_llava_toy_instruction_vlm/`
- `lesson_04_grounding_toy_refexp/`
- `lesson_05_mask_grounding_toy_refexp/`
- `lesson_06_flamingo_toy_interleaved_vlm/`
- `lesson_07_qformer_toy_bridge_vlm/`
- `lesson_08_perceiver_resampler_toy_vlm/`
- `lesson_09_paligemma_toy_siglip_decoder_vlm/`
- `lesson_10_owlvit_toy_open_vocab_detection/`
- `lesson_11_grounded_sam_toy_open_vocab_segmentation/`
- `lesson_12_key_value_ocr_toy_doc_vlm/`
- `lesson_13_video_vlm_toy_temporal_qa/`
- `lesson_14_bmn_toy_temporal_grounding/`
- `lesson_15_2dtan_toy_temporal_grounding/`
- `lesson_16_multiscale_2dtan_toy_temporal_grounding/`

## Why This Track Exists

The local zoo under `dlhub/multimodal/vlm/` is useful for browsing model families and timelines, but it is not the best place to learn the mechanics from scratch. The `tracks/multimodal/` lessons are teaching-first:

- each lesson is independent
- each lesson uses synthetic data so it runs quickly on CPU
- each lesson exposes a small `data.py`, `model.py`, and `train.py` loop
- each lesson highlights one new multimodal idea over the previous lesson

Use the track to understand the ideas. Use the zoo to browse families, variants, and timelines.

## Recommended Progression

### Lesson 1: CLIP-Style Retrieval

Path: `lesson_01_clip_toy_retrieval/`

Core idea:

- image encoder + text encoder
- shared embedding space
- contrastive learning

You should finish this lesson understanding:

- why paired image-text examples can be aligned
- how in-batch negatives work
- why retrieval accuracy is a useful multimodal metric

### Lesson 2: BLIP-Lite Captioning And ITM

Path: `lesson_02_blip_toy_captioning/`

Core idea:

- visual tokens
- text decoding with image conditioning
- image-text matching

You should finish this lesson understanding:

- why multimodal fusion is different from dual-encoder alignment
- how teacher forcing works in image-conditioned text generation
- how ITM complements captioning supervision

### Lesson 3: LLaVA-Lite Visual Instruction VLM

Path: `lesson_03_llava_toy_instruction_vlm/`

Core idea:

- visual tokens as a prefix to a tiny decoder LM
- instruction text plus answer generation
- short visual question answering

You should finish this lesson understanding:

- how a vision projector bridges image features into LM hidden space
- why decoder-only instruction following feels different from BLIP-style fusion
- how multimodal instruction tuning can be simplified into a small local lesson

### Lesson 4: Grounding-Lite Referring Expressions

Path: `lesson_04_grounding_toy_refexp/`

Core idea:

- multi-object scene rendering
- referring-expression conditioning
- grid-cell localization plus box decoding

You should finish this lesson understanding:

- how language can guide spatial localization rather than only retrieval or generation
- why grounding can be decomposed into cell classification plus local box regression
- how region-aware multimodal learning differs from global image-text tasks

### Lesson 5: Mask-Grounding-Lite Referring Expressions

Path: `lesson_05_mask_grounding_toy_refexp/`

Core idea:

- dense low-resolution mask prediction
- text-conditioned spatial fusion
- region grounding instead of only box grounding

You should finish this lesson understanding:

- how referring expressions can supervise per-location mask prediction
- why low-resolution masks are a teaching-friendly bridge to segmentation
- how dense grounded outputs differ from box-only localization

### Lesson 6: Flamingo-Lite Interleaved VLM

Path: `lesson_06_flamingo_toy_interleaved_vlm/`

Core idea:

- interleaved image-text prompting
- support demonstrations plus query in one prompt
- image-aligned token injection inside a decoder-style sequence model

You should finish this lesson understanding:

- how interleaved prompts differ from a pure visual prefix
- why support examples can define the task for a query
- how few-shot multimodal prompting changes the supervision structure

### Lesson 7: Q-Former-Lite Bridge VLM

Path: `lesson_07_qformer_toy_bridge_vlm/`

Core idea:

- learned query tokens
- cross-attention from queries to visual tokens
- compact visual bottleneck before decoding

You should finish this lesson understanding:

- why a query bottleneck differs from direct visual prefixing
- how learned queries can compress visual tokens into a fixed interface
- why this bridge pattern is useful for modular vision-language systems

### Lesson 8: Perceiver-Resampler-Lite VLM

Path: `lesson_08_perceiver_resampler_toy_vlm/`

Core idea:

- multi-view visual token pooling
- fixed latent resampling
- compact decoder prefix after aggressive token compression

You should finish this lesson understanding:

- why resampling matters when visual token counts get large
- how fixed latent arrays differ from direct multi-view decoding
- why multi-view inputs motivate Perceiver-style bottlenecks

### Lesson 9: PaliGemma-Lite SigLIP Decoder VLM

Path: `lesson_09_paligemma_toy_siglip_decoder_vlm/`

Core idea:

- one prompt-native text interface
- decoder-only generation across multiple vision tasks
- SigLIP-style visual tokenization before decoding

You should finish this lesson understanding:

- why captioning, QA, and localization can share one text output interface
- how prompt design can unify multiple tasks without changing heads
- why a strong visual tower plus simple decoder is a useful VLM pattern

### Lesson 10: OWL-ViT-Lite Open-Vocabulary Detection

Path: `lesson_10_owlvit_toy_open_vocab_detection/`

Core idea:

- text-conditioned detection instead of a fixed classifier inventory
- presence plus localization for queries that may be absent
- per-cell fusion with masked localization losses

You should finish this lesson understanding:

- how open-vocabulary detection differs from positive-only grounding
- why presence prediction matters when the query may not exist in the image
- how text-conditioned box prediction can stay simple while still teaching OWL-ViT-like behavior

### Lesson 11: Grounded-SAM-Lite Open-Vocabulary Segmentation

Path: `lesson_11_grounded_sam_toy_open_vocab_segmentation/`

Core idea:

- text-conditioned low-resolution mask prediction
- presence plus segmentation when the query may be absent
- prompt-encoder plus mask-decoder style structure

You should finish this lesson understanding:

- how open-vocabulary segmentation extends lesson 10 from boxes to dense masks
- why positive-only mask losses must be gated by presence
- how a small prompt encoder and mask decoder can explain Grounded-SAM-like behavior

### Lesson 12: Key-Value OCR-Lite Toy Document VLM

Path: `lesson_12_key_value_ocr_toy_doc_vlm/`

Core idea:

- prompt-conditioned document field extraction
- decoder-style OCR as text generation
- missing-field handling through a generated `none` token

You should finish this lesson understanding:

- how document OCR can be framed as multimodal generation instead of box prediction
- why a fixed field inventory still teaches a useful OCR extraction pattern
- how lesson 9's text-generation interface extends naturally into document understanding

### Lesson 13: Video-VLM-Lite Toy Temporal QA

Path: `lesson_13_video_vlm_toy_temporal_qa/`

Core idea:

- short video instead of one image
- temporal aggregation over frame features
- prompt-conditioned generation for appearance and motion questions

You should finish this lesson understanding:

- how a video VLM can stay close to lesson 9 if time is treated as the only new axis
- why frame encoding plus temporal aggregation is a clean teaching bridge into video-language modeling
- how motion questions differ from single-image QA even when the decoder interface stays the same

### Lesson 14: BMN-Lite Toy Temporal Grounding

Path: `lesson_14_bmn_toy_temporal_grounding/`

Core idea:

- text-conditioned temporal localization over a short video
- separate start and end boundary prediction
- upper-triangular proposal scoring for all valid segments

You should finish this lesson understanding:

- why temporal grounding is not the same as video QA
- how boundary supervision and proposal supervision complement each other
- why BMN-style proposal maps are a useful teaching bridge into temporal localization

### Lesson 15: 2D-TAN-Lite Toy Temporal Grounding

Path: `lesson_15_2dtan_toy_temporal_grounding/`

Core idea:

- text-conditioned dense `T x T` segment-map reasoning
- direct scoring over all valid `(start, end)` cells
- lightweight 2D convolution over temporal segment maps

You should finish this lesson understanding:

- why temporal grounding can be framed directly as dense segment-map prediction
- how 2D temporal maps differ from boundary-first proposal scoring
- why segment-level structure can be encoded before final scoring

### Lesson 16: Multi-Scale 2D-TAN-Lite Toy Temporal Grounding

Path: `lesson_16_multiscale_2dtan_toy_temporal_grounding/`

Core idea:

- multi-scale temporal feature pyramid
- one dense segment map per temporal scale
- coarse-to-fine fusion into one final prediction

You should finish this lesson understanding:

- why coarse temporal scales can carry broader context than one fine-scale map alone
- why fine temporal scales still matter for boundary precision
- why deep supervision across scales is a natural extension of lesson 15

## Lesson Matrix

| Lesson | Main Task | Input Form | Model Bridge | Main Loss | Main Metrics |
|---|---|---|---|---|---|
| `lesson_01_clip_toy_retrieval` | image-text retrieval | image + attribute text | shared embedding space | contrastive loss | image-to-text acc, text-to-image acc |
| `lesson_02_blip_toy_captioning` | captioning + ITM | image + sentence caption | visual token fusion into decoder | caption CE + ITM CE | token acc, exact match, ITM acc |
| `lesson_03_llava_toy_instruction_vlm` | visual QA | image + instruction | projected visual prefix into decoder LM | QA token CE | answer token acc, exact match, yes/no acc |
| `lesson_04_grounding_toy_refexp` | referring-expression grounding | image + referring expression | per-cell text-conditioned fusion | cell CE + box regression | cell acc, bbox L1, center acc |
| `lesson_05_mask_grounding_toy_refexp` | mask grounding | image + referring expression | per-location text-conditioned fusion | BCE with logits + dice | mask IoU, dice, foreground acc |
| `lesson_06_flamingo_toy_interleaved_vlm` | interleaved few-shot VLM | support images + query image + prompt | image-slot injection in decoder stream | QA token CE | answer token acc, exact match |
| `lesson_07_qformer_toy_bridge_vlm` | query-bottleneck visual QA | image + question | learned query bridge into decoder LM | QA token CE | answer token acc, exact match, yes/no acc |
| `lesson_08_perceiver_resampler_toy_vlm` | multi-view resampled visual QA | full scene + crop views + question | Perceiver-style latent resampler into decoder LM | QA token CE | answer token acc, exact match, yes/no acc |
| `lesson_09_paligemma_toy_siglip_decoder_vlm` | prompt-native multitask VLM | image + task prompt | SigLIP-style vision tokens into decoder LM | QA token CE | answer token acc, exact match, yes/no acc |
| `lesson_10_owlvit_toy_open_vocab_detection` | open-vocabulary detection | image + text query | per-cell text-conditioned detector | presence BCE + cell CE + box regression | presence acc, bbox L1, center acc |
| `lesson_11_grounded_sam_toy_open_vocab_segmentation` | open-vocabulary segmentation | image + text query | prompt-conditioned mask decoder | presence BCE + mask BCE + dice | presence acc, mask IoU, dice, foreground acc |
| `lesson_12_key_value_ocr_toy_doc_vlm` | key-value document OCR | document image + field prompt | visual prefix into decoder LM | answer token CE | answer token acc, exact match, present acc |
| `lesson_13_video_vlm_toy_temporal_qa` | temporal video QA | short video + question prompt | frame encoder + temporal aggregator into decoder LM | answer token CE | answer token acc, exact match, yes/no acc |
| `lesson_14_bmn_toy_temporal_grounding` | temporal grounding | short video + event query | query-conditioned temporal encoder + BMN-lite proposal map | start BCE + end BCE + proposal MSE | start acc, end acc, mean tIoU, R@1 IoU=0.5 |
| `lesson_15_2dtan_toy_temporal_grounding` | temporal grounding | short video + event query | dense 2D temporal segment map + 2D conv scoring | masked map MSE | mean tIoU, R@1 IoU=0.5, R@1 IoU=0.7 |
| `lesson_16_multiscale_2dtan_toy_temporal_grounding` | temporal grounding | short video + event query | multi-scale dense temporal segment maps + fused scoring | fused map MSE + auxiliary masked map MSE | mean tIoU, R@1 IoU=0.5, R@1 IoU=0.7 |

## Quick Start

List the track:

```bash
python scripts/run_lesson.py multimodal --list
```

Dry-run a specific lesson:

```bash
python scripts/run_lesson.py multimodal lesson_03_llava_toy_instruction_vlm --dry-run
```

Smoke run lesson 1:

```bash
python -m tracks.multimodal.lesson_01_clip_toy_retrieval.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_clip
```

Smoke run lesson 2:

```bash
python -m tracks.multimodal.lesson_02_blip_toy_captioning.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_blip
```

Smoke run lesson 3:

```bash
python -m tracks.multimodal.lesson_03_llava_toy_instruction_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_llava
```

Smoke run lesson 4:

```bash
python -m tracks.multimodal.lesson_04_grounding_toy_refexp.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_grounding
```

Smoke run lesson 5:

```bash
python -m tracks.multimodal.lesson_05_mask_grounding_toy_refexp.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_mask_grounding
```

Smoke run lesson 6:

```bash
python -m tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_flamingo
```

Smoke run lesson 7:

```bash
python -m tracks.multimodal.lesson_07_qformer_toy_bridge_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_qformer
```

Smoke run lesson 8:

```bash
python -m tracks.multimodal.lesson_08_perceiver_resampler_toy_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_perceiver
```

Smoke run lesson 9:

```bash
python -m tracks.multimodal.lesson_09_paligemma_toy_siglip_decoder_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_paligemma
```

Smoke run lesson 10:

```bash
python -m tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_owlvit
```

Smoke run lesson 11:

```bash
python -m tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_grounded_sam
```

Smoke run lesson 12:

```bash
python -m tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_key_value_ocr
```

Smoke run lesson 13:

```bash
python -m tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_video_vlm
```

Smoke run lesson 14:

```bash
python -m tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_bmn
```

Smoke run lesson 15:

```bash
python -m tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_2dtan
```

Smoke run lesson 16:

```bash
python -m tracks.multimodal.lesson_16_multiscale_2dtan_toy_temporal_grounding.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_multiscale_2dtan
```

## How To Study Each Lesson

Recommended order inside every lesson:

1. Read `README.md` to understand the task and output artifacts.
2. Read `data.py` first so the supervision target is obvious.
3. Read `model.py` second so the architectural change is easy to see.
4. Read `train.py` last to connect model outputs to loss and metrics.
5. Run a smoke command on CPU and inspect `outputs/.../samples.jsonl`.

If you want the shortest path through the track:

1. Run lesson 1 once and inspect retrieval metrics.
2. Run lesson 2 once and compare generation plus ITM outputs.
3. Run lesson 3 once and inspect short-answer instruction predictions.
4. Run lesson 4 once and compare target cells and decoded boxes in `samples.jsonl`.
5. Run lesson 5 once and compare target and predicted foreground ratios in `samples.jsonl`.
6. Run lesson 6 once and inspect how support examples shape the query answer in `samples.jsonl`.
7. Run lesson 7 once and compare its query bottleneck against lesson 3's direct visual prefix.
8. Run lesson 8 once and compare multi-view resampling against lesson 7's single-image query bridge.
9. Run lesson 9 once and compare how one decoder handles caption, QA, and text localization prompts.
10. Run lesson 10 once and compare absent-query detection against lesson 4's always-positive grounding setup.
11. Run lesson 11 once and compare absent-query segmentation against lesson 5's always-positive mask grounding setup.
12. Run lesson 12 once and compare document field extraction against lesson 9's object-centric prompt generation.
13. Run lesson 13 once and compare temporal QA against lesson 3 or lesson 9 to isolate what the temporal aggregator changes.
14. Run lesson 14 once and compare temporal grounding against lesson 13 to separate "answering over time" from "localizing in time".
15. Run lesson 15 once and compare dense segment-map reasoning against lesson 14's boundary-first BMN-lite formulation.
16. Run lesson 16 once and compare multi-scale fused localization against lesson 15's single-scale temporal map.

## Lessons Versus Zoo

Use the lessons when you want:

- teaching-sized code
- runnable local experiments
- a clean progression from alignment to generation to instruction following

Use the zoo when you want:

- family browsing by year
- multiple VLM variants and timelines
- recommendation profiles
- quick local smoke models that resemble paper families

The important distinction is:

- `tracks/multimodal/` explains ideas
- `dlhub/multimodal/vlm/` organizes families

## Output Convention

Each lesson writes to:

- `outputs/multimodal/<lesson_name>/<run_name>/`

Typical artifacts:

- `config.json`
- `vocab.json`
- `metrics.jsonl`
- `samples.jsonl`
- `logs/train.log`
- `checkpoints/checkpoint.pt`

## Next Steps

After these fifteen lessons, the next useful directions are:

- extend lesson 3 with more instruction types
- extend lesson 4 and lesson 5 toward phrase grounding, region ranking, or open-vocabulary segmentation
- extend lesson 6 toward gated cross-attention, larger context windows, or mixed-image dialogue
- extend lesson 7 toward frozen backbones, stronger query formers, or retrieval-augmented vision bridges
- extend lesson 8 toward video resampling, temporal latents, or larger multi-image contexts
- extend lesson 9 toward OCR, detection-as-text, or richer structured generation
- extend lesson 10 toward multi-query ranking, larger vocabularies, or phrase-level open-vocabulary detection
- extend lesson 11 toward interactive point prompts, higher-resolution refinement, or open-vocabulary phrase segmentation
- extend lesson 12 toward multi-field extraction, longer OCR sequences, or free-form document QA
- extend lesson 13 toward multi-object temporal grounding, longer clips, speed questions, or interleaved video-language prompting
- extend lesson 14 toward multi-moment grounding, longer clips, proposal ranking heads, or more natural language event descriptions
- extend lesson 15 toward stronger query-conditioned map refinement or alternative 2D map builders
- extend lesson 16 toward multi-moment retrieval, longer clips, or stronger query-conditioned multi-scale fusion
- compare the teaching lessons against the local zoo families in `dlhub/multimodal/vlm/`
