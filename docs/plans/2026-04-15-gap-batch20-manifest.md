# Batch 20 Track-Fill Manifest

Batch 20 continues the fixed `10 worktree + 10 lane` loop, and this time it can stay entirely on
direct mappings from the user tag pool. The selection extends the teaching tracks into RLHF-style
optimization, GAN-conditioned generation, embodied/multimodal reasoning, and restoration/fusion
vision tasks while preserving the same track-first balance.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- user tag pool first, then neighbor fill where the tag pool had no clean direct lesson mapping

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch20 track-fill manifest`
- integration branch: `batch200-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch200-span-corruption` | `.worktrees/batch200-span-corruption` | `tracks/llm` | `span_corruption_llm_lesson` | `tracks/llm/lesson_08_compact_span_corruption/`, `tests/test_tracks_llm_span_corruption.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_span_corruption.py`<br>`python scripts/run_lesson.py llm lesson_08_compact_span_corruption --dry-run` |
| L02 | `batch200-rlhf-ppo` | `.worktrees/batch200-rlhf-ppo` | `tracks/llm` | `rlhf_ppo_llm_lesson` | `tracks/llm/lesson_09_compact_rlhf_ppo/`, `tests/test_tracks_llm_rlhf_ppo.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_rlhf_ppo.py`<br>`python scripts/run_lesson.py llm lesson_09_compact_rlhf_ppo --dry-run` |
| L03 | `batch200-conditional-gan` | `.worktrees/batch200-conditional-gan` | `tracks/generative` | `conditional_gan_lesson` | `tracks/generative/lesson_09_compact_conditional_gan/`, `tests/test_tracks_generative_conditional_gan.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_conditional_gan.py`<br>`python scripts/run_lesson.py generative lesson_09_compact_conditional_gan --dry-run` |
| L04 | `batch200-diffusion-image-editing` | `.worktrees/batch200-diffusion-image-editing` | `tracks/generative` | `diffusion_image_editing_lesson` | `tracks/generative/lesson_10_compact_diffusion_image_editing/`, `tests/test_tracks_generative_diffusion_image_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_image_editing.py`<br>`python scripts/run_lesson.py generative lesson_10_compact_diffusion_image_editing --dry-run` |
| L05 | `batch200-embodied-question-answering` | `.worktrees/batch200-embodied-question-answering` | `tracks/multimodal` | `embodied_question_answering_lesson` | `tracks/multimodal/lesson_23_embodied_question_answering/`, `tests/test_tracks_multimodal_embodied_question_answering.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_embodied_question_answering.py`<br>`python scripts/run_lesson.py multimodal lesson_23_embodied_question_answering --dry-run` |
| L06 | `batch200-multimodal-reasoning` | `.worktrees/batch200-multimodal-reasoning` | `tracks/multimodal` | `multimodal_reasoning_lesson` | `tracks/multimodal/lesson_24_multimodal_reasoning/`, `tests/test_tracks_multimodal_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_24_multimodal_reasoning --dry-run` |
| L07 | `batch200-reflection-removal` | `.worktrees/batch200-reflection-removal` | `tracks/vision` | `reflection_removal_lesson` | `tracks/vision/lesson_24_synthetic_reflection_removal/`, `tests/test_tracks_vision_reflection_removal.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_reflection_removal.py`<br>`python scripts/run_lesson.py vision lesson_24_synthetic_reflection_removal --dry-run` |
| L08 | `batch200-image-fusion` | `.worktrees/batch200-image-fusion` | `tracks/vision` | `image_fusion_lesson` | `tracks/vision/lesson_25_synthetic_image_fusion/`, `tests/test_tracks_vision_image_fusion.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_image_fusion.py`<br>`python scripts/run_lesson.py vision lesson_25_synthetic_image_fusion --dry-run` |
| L09 | `batch200-contrastive-sentence-embedding` | `.worktrees/batch200-contrastive-sentence-embedding` | `tracks/nlp` | `contrastive_sentence_embedding_lesson` | `tracks/nlp/lesson_14_compact_contrastive_sentence_embedding/`, `tests/test_tracks_nlp_contrastive_sentence_embedding.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_contrastive_sentence_embedding.py`<br>`python scripts/run_lesson.py nlp lesson_14_compact_contrastive_sentence_embedding --dry-run` |
| L10 | `batch200-cross-encoder-reranking` | `.worktrees/batch200-cross-encoder-reranking` | `tracks/nlp` | `cross_encoder_reranking_lesson` | `tracks/nlp/lesson_15_compact_cross_encoder_reranking/`, `tests/test_tracks_nlp_cross_encoder_reranking.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_cross_encoder_reranking.py`<br>`python scripts/run_lesson.py nlp lesson_15_compact_cross_encoder_reranking --dry-run` |

## Integration Branch Ownership

`batch200-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch200-span-corruption`
2. `batch200-rlhf-ppo`
3. `batch200-conditional-gan`
4. `batch200-diffusion-image-editing`
5. `batch200-embodied-question-answering`
6. `batch200-multimodal-reasoning`
7. `batch200-reflection-removal`
8. `batch200-image-fusion`
9. `batch200-contrastive-sentence-embedding`
10. `batch200-cross-encoder-reranking`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
