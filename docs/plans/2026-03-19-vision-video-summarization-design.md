# Vision Video Summarization Design

**Goal**

Add a local, toy-first video summarization algorithm family to DL-Hub with:
- local video summarization model families
- a unified zoo + CLI
- fast smoke tests that run on CPU with no downloads

This mirrors the existing repository patterns used by:
- `dlhub.vision.mot_zoo`
- `dlhub.vision.action_recognition_zoo`
- `dlhub.vision.style_transfer_zoo`

---

## Scope

This feature targets extractive video summarization rather than text generation.

Input:
- short video tensor with shape `(B, T, C, H, W)`

Output:
- `scores`: frame importance scores with shape `(B, T)`
- `summary_mask`: binary or soft summary mask with shape `(B, T)`

Optional outputs may include:
- `features`
- `segment_scores`
- `recon_loss`
- `discriminator_logits`

The family is meant to teach how video summarization models score frames or segments, not to ship full benchmark reproductions.

---

## Package Layout

Add:
- `dlhub/vision/video_summarization/`
- `dlhub/vision/video_summarization_zoo.py`
- `scripts/video_summarization_zoo.py`
- `tests/test_dlhub_vision_video_summarization_zoo.py`

Initial local families:
- `dsn`: reinforcement-learning style scoring baseline (toy deterministic scorer)
- `sum_gan`: adversarial summarization (toy generator + discriminator)
- `cycle_sum`: cycle-consistent summary/reconstruction (toy)
- `vasnet`: self-attention frame scorer
- `dsnet`: detect-to-summarize segment proposal scorer
- `ca_sum`: content-attention summarizer
- `pgl_sum`: local-global hybrid attention summarizer
- `mhscnet`: shot-aware multi-scale conv summarizer (toy unimodal adaptation)
- `tac_sum`: temporal-aware clustering summarizer (toy training-free adaptation)
- `csta`: CNN-based spatiotemporal attention summarizer
- `fulltransnet`: full transformer encoder-decoder summarizer
- `summdiff`: diffusion-style score generation summarizer
- `qfvs_memnet`: query-focused memory-network summarizer with internal prompt fallback
- `videograph`: graph message-passing frame summarizer
- `lgrln`: language-guided relation learning summarizer with optional query conditioning
- `intentvizor`: intent-guided interactive summarizer with ego-graph reasoning
- `maam`: multi-annotation attention summarizer with latent annotator aggregation
- `checkmate`: temporal encapsulation summarizer with mutual context checking
- `viewpoint_sum`: viewpoint-aware summarizer with latent viewpoint prototypes
- `progressive_ssl`: progressive self-supervised summarizer with concept prompt refinement
- `llm_pretrain`: LLM-oracle-inspired pretraining summarizer with prompt-token distillation
- `contrast_sum`: unsupervised contrastive summarizer with paired temporal views
- `mc_vsa`: multi-concept attention summarizer with concept-conditioned scoring
- `multi_stream_sum`: multi-stream summarizer with appearance-motion-context fusion
- `personalized_ranker`: personalized summarizer with multiple pairwise rankers
- `sem_reward_rl`: semantic-reward reinforcement-learning summarizer
- `dp_dtw_sum`: prototype-DTW action-aware summarizer
- `hsa_rnn`: hierarchical structure-adaptive RNN summarizer
- `clip_it`: language-guided multimodal transformer summarizer
- `videosage`: sparse graph representation learning summarizer
- `pfmn`: past-future memory-network summarizer
- `a2summ`: align-and-attend multimodal dual-contrastive summarizer
- `iterative_gan`: iterative simplified GAN summarizer

Each family provides three variants:
- `*_tiny`
- `*_small`
- `*_base`

Zoo prefix:
- `vsum:<variant>`

Current expanded local coverage:
- 33 families
- 99 arches

---

## Model Contract

Every family exposes:
- `_VARIANTS`
- `build_<family>_video_summarizer(...)`

Every model supports:
- `model(video)` where `video` is `(B, T, C, H, W)`

Every model returns a dict with:
- `scores`
- `summary_mask`

This keeps the contract uniform across attention, adversarial, proposal-based, and reconstruction-style methods.

---

## Shared Components

Add a small `_common.py` with:
- input validation for `(B, T, C, H, W)`
- lightweight frame encoder
- temporal encoder blocks
- helpers to convert scores into summary masks
- optional segment pooling helpers for proposal-style models

The shared code should stay small and generic. Do not build a monolithic framework.

---

## CLI

Add `scripts/video_summarization_zoo.py` with:
- `--list`
- `--search`
- `--limit`
- `--smoke`

Smoke mode should:
- create a random video tensor
- build one local summarizer
- run forward
- print a concise output summary

---

## Testing

Add focused tests for:
- listing arches
- building several representative families
- running forward smoke on random video
- CLI `--list`
- CLI `--smoke`

Keep runtime low. No lesson is required in the first pass.
