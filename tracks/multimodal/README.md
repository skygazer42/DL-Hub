# Multimodal Track

Goal: learn the core progression of modern vision-language modeling through small, runnable teaching implementations.

This track is intentionally ordered as a fifty-eight-step path:

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
17. retrieve the matching text for a short video by adding temporal contrastive alignment
18. adapt a frozen multimodal retriever by learning only soft prompt tokens
19. align short audio clips with text labels for event understanding
20. fuse synchronized audio and video streams for clip-level cross-modal learning
21. retrieve the matching clip by conditioning language on audio evidence inside a temporal multimodal scene
22. localize when a queried event happens by fusing audio, video, and text over time
23. answer a question from egocentric observations plus navigation state in an embodied scene
24. reason jointly over image evidence and supporting facts to choose the correct answer
25. navigate from visual observations plus a language instruction by fusing scene evidence with navigation state
26. rerank matched image-text candidates with a cross-encoder style fusion scorer
27. read short scene text directly from an image with a compact vision-language recognizer
28. answer simple document questions by combining OCR tokens, layout cues, and a textual query
29. reason over human and object regions jointly with a language relation query
30. estimate a gaze target from image evidence, head-location cues, and short textual context
31. retrieve the matching person image from an attribute-style text query with identity-aware alignment
32. localize when a described action occurs in a short clip from fused video frames and text cues
33. predict pedestrian attributes from a person image against a text attribute inventory
34. classify a short clip against textual action descriptions with compact video-language alignment
35. classify facial expressions from compact face evidence against short emotion prompts
36. judge whether a face is authentic or spoofed from visual artifacts plus a short textual cue
37. match a face against candidate identity prompts with compact image-text alignment
38. verify whether two face observations belong to the same identity with multimodal pair reasoning
39. reason about a facial attribute from image evidence plus a short attribute query
40. judge whether a short face caption is grounded in the observed face image
41. reason about whether a face is lightly or heavily occluded from image evidence plus a short query
42. ground a queried facial region to a normalized image box from fused image-text evidence
43. reason about requested facial landmarks from image evidence plus a short landmark query
44. reason about face-part parsing from image evidence plus a short region query
45. regress canonical facial landmarks from image evidence plus a short alignment query
46. localize a face box from image evidence plus a short detection query
47. retrieve the matching face identity from a small gallery with image-text alignment
48. regress normalized yaw, pitch, and roll from face evidence plus a short pose query
49. regress a face-centered gaze target from image evidence plus a short gaze query
50. regress a compact person-pose state from image evidence plus a short pose query
51. regress a compact hand-pose state from image evidence plus a short hand-pose query
52. classify a compact gesture state from image evidence plus a short gesture query
53. classify finger count from grayscale hand evidence plus a short count query
54. classify handedness from grayscale hand evidence plus a short handedness query
55. reason about palm orientation from grayscale hand evidence plus a short orientation query
56. reason about sign-digit identity from grayscale hand evidence plus a short digit query
57. regress normalized finger spread from grayscale hand evidence plus a short spread query
58. classify thumb position from grayscale hand evidence plus a short thumb-position query

That progression maps directly to the fifty-eight current lessons:

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
- `lesson_17_video_text_retrieval/`
- `lesson_18_prompt_learning_vlm/`
- `lesson_19_audio_text_understanding/`
- `lesson_20_audio_visual_learning/`
- `lesson_21_audio_grounded_retrieval/`
- `lesson_22_audio_visual_event_localization/`
- `lesson_23_embodied_question_answering/`
- `lesson_24_multimodal_reasoning/`
- `lesson_25_vision_language_navigation/`
- `lesson_26_image_text_reranking/`
- `lesson_27_scene_text_vlm_recognition/`
- `lesson_28_document_vlm_reasoning/`
- `lesson_29_human_object_interaction_reasoning/`
- `lesson_30_vision_language_gaze_estimation/`
- `lesson_31_person_search_attribute_retrieval/`
- `lesson_32_video_text_action_localization/`
- `lesson_33_pedestrian_attribute_recognition/`
- `lesson_34_video_text_action_recognition/`
- `lesson_35_face_expression_vlm_recognition/`
- `lesson_36_face_anti_spoof_vlm_reasoning/`
- `lesson_37_face_identity_vlm_recognition/`
- `lesson_38_face_verification_vlm_reasoning/`
- `lesson_39_face_attribute_vlm_reasoning/`
- `lesson_40_face_caption_vlm_grounding/`
- `lesson_41_face_occlusion_vlm_reasoning/`
- `lesson_42_face_region_grounding_vlm/`
- `lesson_43_face_landmark_vlm_reasoning/`
- `lesson_44_face_parsing_vlm_reasoning/`
- `lesson_45_face_alignment_vlm_reasoning/`
- `lesson_46_face_detection_vlm_reasoning/`
- `lesson_47_face_retrieval_vlm_reasoning/`
- `lesson_48_face_pose_vlm_reasoning/`
- `lesson_49_face_gaze_vlm_reasoning/`
- `lesson_50_person_pose_vlm_reasoning/`
- `lesson_51_hand_pose_vlm_reasoning/`
- `lesson_52_gesture_vlm_reasoning/`
- `lesson_53_finger_count_vlm_reasoning/`
- `lesson_54_handedness_vlm_reasoning/`
- `lesson_55_palm_orientation_vlm_reasoning/`
- `lesson_56_sign_digit_vlm_reasoning/`
- `lesson_57_finger_spread_vlm_reasoning/`
- `lesson_58_thumb_position_vlm_reasoning/`

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

### Lesson 17: Video-Text Retrieval

Path: `lesson_17_video_text_retrieval/`

Core idea:

- frame encoder plus temporal pooling
- shared video-text embedding space
- symmetric contrastive retrieval over short clips

You should finish this lesson understanding:

- how CLIP-style retrieval extends from one image to a short video
- why temporal pooling is a simple but useful bridge into video-language alignment
- how retrieval metrics expose cross-modal alignment quality directly

### Lesson 18: Prompt Learning VLM

Path: `lesson_18_prompt_learning_vlm/`

Core idea:

- frozen image encoder and frozen text encoder
- learnable soft prompt tokens on the text side
- CoOp-style adaptation without full-model finetuning

You should finish this lesson understanding:

- how soft prompts can adapt a multimodal model while keeping most weights frozen
- why prompt learning is a lightweight alternative to end-to-end finetuning
- how prompt tokens shift text embeddings to improve retrieval

### Lesson 19: Audio-Text Understanding

Path: `lesson_19_audio_text_understanding/`

Core idea:

- audio encoder plus text encoder in a shared event embedding space
- waveform-like clips paired with natural-language event descriptions
- one lightweight classification head beside contrastive alignment

You should finish this lesson understanding:

- how audio-text retrieval reuses the CLIP-style recipe on top of synthetic audio features
- why event descriptions are a clean supervision bridge between retrieval and classification
- how one shared embedding space can support both alignment and event prediction

### Lesson 20: Audio-Visual Learning

Path: `lesson_20_audio_visual_learning/`

Core idea:

- synchronized audio tokens and visual frame tokens
- lightweight fusion for clip-level event understanding
- robustness checks against missing or weak single-modality evidence

You should finish this lesson understanding:

- how synchronized audio and video complement each other for event recognition
- why a small fusion block is enough to expose cross-modal gains in a toy setting
- how clip-level multimodal classification differs from text-conditioned retrieval or generation

### Lesson 21: Audio-Grounded Retrieval

Path: `lesson_21_audio_grounded_retrieval/`

Core idea:

- language query plus audio-guided temporal retrieval
- aligned audio and frame tokens compressed into one clip embedding
- retrieval over short multimodal scenes rather than over one static clip label

You should finish this lesson understanding:

- how retrieval changes once the query must key into both text semantics and audio evidence
- why temporal clip retrieval is a useful bridge between clip-level classification and frame-level localization
- how one shared embedding space can support retrieval across fused audio-video scenes

### Lesson 22: Audio-Visual Event Localization

Path: `lesson_22_audio_visual_event_localization/`

Core idea:

- short audio-video windows plus a text event query
- fused temporal tokens that score when the queried event occurs
- frame-level localization instead of clip-level retrieval

You should finish this lesson understanding:

- how event localization differs from clip retrieval even when the same modalities are present
- why a text query is useful for selecting one event out of multiple audio-visual cues
- how temporal saliency prediction closes the gap between retrieval and grounding in multimodal teaching code

### Lesson 23: Embodied Question Answering

Path: `lesson_23_embodied_question_answering/`

Core idea:

- egocentric observations plus a small navigation-state summary
- question answering grounded in embodied scene context
- multimodal fusion over scene tokens, state tokens, and question tokens

You should finish this lesson understanding:

- how embodied QA differs from static-image VQA because scene state matters
- why navigation metadata is a useful bridge between perception and reasoning
- how a small fused encoder can answer grounded questions without a large world model

### Lesson 24: Multimodal Reasoning

Path: `lesson_24_multimodal_reasoning/`

Core idea:

- image evidence plus a short sequence of supporting facts
- reasoning over multiple candidate answers instead of free-form generation
- joint visual-text pooling for answer selection

You should finish this lesson understanding:

- how multimodal reasoning differs from retrieval or captioning once evidence must be combined
- why structured fact tokens are a simple teaching scaffold for multi-hop reasoning
- how answer classification exposes whether the model used both image and text evidence

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
| `lesson_17_video_text_retrieval` | video-text retrieval | short video + text query | frame encoder + temporal pooling into shared embedding space | symmetric contrastive loss | top-1 retrieval acc, recall@3 |
| `lesson_18_prompt_learning_vlm` | prompt learning retrieval | image + learnable prompt text | frozen encoders + soft text prompts | contrastive loss on prompted text embeddings | top-1 retrieval acc, prompt-only adaptation gap |
| `lesson_19_audio_text_understanding` | audio-text retrieval + event classification | waveform clip + event text | audio encoder + text encoder in shared event space | symmetric contrastive loss + event CE | retrieval acc, event acc |
| `lesson_20_audio_visual_learning` | audio-visual event understanding | clip frames + synchronized audio | audio branch + visual branch + fused classifier | event CE | fused acc, audio-only gap, visual-only gap |
| `lesson_21_audio_grounded_retrieval` | audio-grounded retrieval | short multimodal scene + query text | fused audio-video scene encoder into shared retrieval space | symmetric contrastive loss | clip retrieval acc, recall@3 |
| `lesson_22_audio_visual_event_localization` | audio-visual event localization | short clip + audio + event query text | fused temporal encoder + localization scorer | frame BCE + temporal smoothness | frame acc, localization IoU, peak hit rate |
| `lesson_23_embodied_question_answering` | embodied question answering | egocentric scene + navigation state + question | scene/state fusion encoder into answer classifier | answer CE | answer acc, macro F1 |
| `lesson_24_multimodal_reasoning` | multimodal reasoning | image + supporting facts + question + answer choices | visual-text reasoning encoder into choice scorer | answer CE | answer acc, choice margin |
| `lesson_25_vision_language_navigation` | vision-language navigation | egocentric observation + route instruction + navigation state | visual-state instruction fusion into action policy | action CE | action acc, path success |
| `lesson_26_image_text_reranking` | image-text reranking | image + query text + candidate captions | cross-encoder fusion scorer over paired candidates | pairwise ranking BCE | rerank acc, mean reciprocal rank |
| `lesson_27_scene_text_vlm_recognition` | scene-text recognition | scene image + text prompt | visual token encoder into compact decoder recognizer | token CE | token acc, exact match |
| `lesson_28_document_vlm_reasoning` | document reasoning | document image + question | OCR/layout fusion encoder into answer classifier | answer CE | answer acc, macro F1 |
| `lesson_29_human_object_interaction_reasoning` | human-object interaction reasoning | human region + object region + relation query | region fusion encoder into interaction classifier | answer CE | answer acc, macro F1 |
| `lesson_30_vision_language_gaze_estimation` | gaze estimation | image + head location + short prompt | visual-language fusion into gaze point + heatmap heads | point regression + heatmap BCE | point L1, heatmap IoU, hit acc |
| `lesson_31_person_search_attribute_retrieval` | person-search attribute retrieval | person image + attribute text query | dual encoder into shared identity-aware embedding space | symmetric contrastive loss | image-to-text acc, text-to-image acc, recall@3 |
| `lesson_32_video_text_action_localization` | video-text action localization | short clip features + action query | query-conditioned temporal encoder into start/end regressors | start CE + end CE + span L1 | start acc, end acc, mean tIoU |
| `lesson_33_pedestrian_attribute_recognition` | pedestrian attribute recognition | person image + attribute text inventory | visual encoder + text attribute encoder into multi-label scorer | BCE with logits over attributes | attribute acc, macro F1, exact-match rate |
| `lesson_34_video_text_action_recognition` | video-text action recognition | short clip features + action text labels | temporal video encoder + text label encoder into shared classifier | action CE + alignment CE | action acc, retrieval acc |

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

Smoke run lesson 17:

```bash
python -m tracks.multimodal.lesson_17_video_text_retrieval.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_video_text_retrieval
```

Smoke run lesson 18:

```bash
python -m tracks.multimodal.lesson_18_prompt_learning_vlm.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_prompt_learning
```

Smoke run lesson 19:

```bash
python -m tracks.multimodal.lesson_19_audio_text_understanding.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_audio_text
```

Smoke run lesson 20:

```bash
python -m tracks.multimodal.lesson_20_audio_visual_learning.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_audio_visual
```

Smoke run lesson 21:

```bash
python -m tracks.multimodal.lesson_21_audio_grounded_retrieval.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_audio_grounded_retrieval
```

Smoke run lesson 22:

```bash
python -m tracks.multimodal.lesson_22_audio_visual_event_localization.train --device cpu --epochs 1 --max-train-batches 2 --max-eval-batches 1 --run-name smoke_audio_visual_localization
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
17. Run lesson 17 once and compare retrieval over full clips against lesson 13's question answering over time.
18. Run lesson 18 once and compare soft-prompt adaptation against lesson 17's full retriever training.
19. Run lesson 19 once and compare audio-text retrieval against the earlier image-text and video-text retrieval setups.
20. Run lesson 20 once and inspect how synchronized audio shifts predictions relative to visual-only clip recognition.
21. Run lesson 21 once and compare clip retrieval with explicit audio evidence against lesson 20's clip classification.
22. Run lesson 22 once and compare frame-level localization against lesson 21's clip-level retrieval objective.

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
