# Multimodal Lesson 07 Q-Former-Lite Design

**Date:** 2026-03-14

**Goal:** Add `tracks/multimodal/lesson_07_qformer_toy_bridge_vlm` as an independent teaching lesson for query-bottleneck vision-language bridging in the style of BLIP-2 and Q-Former systems.

## Problem

The multimodal track already covers:

- lesson 1: dual-encoder alignment
- lesson 2: BLIP-lite fusion
- lesson 3: LLaVA-lite visual prefix instruction tuning
- lesson 4: box grounding
- lesson 5: mask grounding
- lesson 6: Flamingo-lite interleaved prompting

What is still missing is the bridge pattern where a small learnable query module compresses visual tokens into a fixed number of latent tokens before passing them to a language model. That query bottleneck is a different idea from both:

- lesson 3, where visual tokens are projected directly into the decoder stream
- lesson 6, where images are aligned to slots inside an interleaved prompt

## Candidate Directions

### Option A: Reuse direct visual prefixing

- encode visual tokens
- project them into LM hidden space
- prepend them to the text sequence

Pros:

- minimal implementation

Cons:

- already covered by lesson 3
- does not teach the query bottleneck

### Option B: Collapse the image to one pooled vector

- encode the image
- average pool to one vector
- initialize a decoder with that vector

Pros:

- extremely simple

Cons:

- too weak pedagogically
- hides the whole reason Q-Former exists

### Option C: Learnable query tokens cross-attend to visual tokens

- encode image patches or spatial tokens
- introduce a small set of learned query tokens
- let queries read from visual tokens through cross-attention
- feed the resulting query states into a tiny decoder LM

Pros:

- teaches the key BLIP-2/Q-Former idea cleanly
- keeps sequence length under control
- creates a direct contrast with lesson 3

Cons:

- slightly more moving parts than a direct prefix

## Recommendation

Choose Option C.

This is the simplest formulation that still honestly teaches the core idea: a compact learned query interface between the vision encoder and the language model.

## Task Formulation

Keep the task simple and stable by reusing the single-image QA setting:

- input: one synthetic image plus one short question
- output: one short answer

Examples:

- `what color is the object`
- `what shape is the object`
- `where is the object`
- `is the object red`

This keeps the dataset familiar, so the lesson difference is architectural rather than data-related.

## Data Design

The lesson should define its own teaching dataset, even if it resembles lesson 3.

Each record should include:

- `image`
- `question_ids`
- `input_ids`
- `labels`
- `attention_mask`
- `question_type`
- `question_text`
- `answer_text`

Recommended defaults:

- `image_size = 16`
- `max_text_length = 12`
- single object per image

The supervision target should only score the answer tokens, not the question prefix.

## Model Design

### Vision encoder

- small CNN
- output spatial visual tokens rather than a single pooled vector

### Q-Former-lite bridge

- create `num_query_tokens` learned query embeddings
- repeat them for the batch
- run one or two lightweight cross-attention/update blocks where queries read visual tokens

The update block can be simplified for teaching:

- attention from query tokens to visual tokens
- residual update with a small MLP
- optional layer norm

### Decoder LM

- project query states into the decoder hidden space
- use them as a short prefix to a tiny GRU-based decoder LM
- decode the question-plus-answer sequence

This keeps the architecture visibly different from lesson 3:

- lesson 3: all visual tokens directly prefix the LM
- lesson 7: learned queries compress vision before the LM sees it

## Losses and Metrics

Training loss:

- token cross entropy over answer tokens only

Metrics:

- answer token accuracy
- answer exact match
- yes/no accuracy

## Training Script

Follow the same conventions as lessons 1-6:

- CLI args for data size, image size, text length, and query-token count
- `outputs/multimodal/lesson_07_qformer_toy_bridge_vlm/<run_name>/`
- `config.json`, `vocab.json`, `metrics.jsonl`, `samples.jsonl`, logs, checkpoint

Sample logging should include:

- question type
- question text
- answer gt
- answer pred

## Success Criteria

Lesson 7 is complete when:

- it is discoverable through `scripts/run_lesson.py`
- batch/data tests pass
- model/loss tests pass
- CPU smoke training passes
- the track README includes lesson 7
- the lesson clearly demonstrates query bottleneck bridging
