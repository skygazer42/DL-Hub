# Batch 17 Track-Fill Manifest

Batch 17 is the first full 10-lane batch after the one-time Batch 16 half-batch exception. It
switches from `dlhub/` direction packages to track-first lesson filling so the active teaching
surfaces stop lagging behind the expanded model-zoo coverage.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- user tag pool first, then neighbor fill where the tag pool had no clean direct lesson mapping

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch17 track-fill manifest`
- integration branch: `batch170-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch170-chat-sft` | `.worktrees/batch170-chat-sft` | `tracks/llm` | `conversational_llm_lesson` | `tracks/llm/lesson_02_compact_chat_sft/`, `tests/test_tracks_llm_chat_sft.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_chat_sft.py`<br>`python scripts/run_lesson.py llm lesson_02_compact_chat_sft --dry-run` |
| L02 | `batch170-mamba-llm` | `.worktrees/batch170-mamba-llm` | `tracks/llm` | `mamba_sequence_lesson` | `tracks/llm/lesson_03_compact_mamba_language_model/`, `tests/test_tracks_llm_mamba.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_mamba.py`<br>`python scripts/run_lesson.py llm lesson_03_compact_mamba_language_model --dry-run` |
| L03 | `batch170-diffusion-lesson` | `.worktrees/batch170-diffusion-lesson` | `tracks/generative` | `diffusion_lesson` | `tracks/generative/lesson_03_compact_diffusion_mnist/`, `tests/test_tracks_generative_diffusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion.py`<br>`python scripts/run_lesson.py generative lesson_03_compact_diffusion_mnist --dry-run` |
| L04 | `batch170-latent-diffusion-lesson` | `.worktrees/batch170-latent-diffusion-lesson` | `tracks/generative` | `latent_diffusion_lesson` | `tracks/generative/lesson_04_compact_latent_diffusion/`, `tests/test_tracks_generative_latent_diffusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_latent_diffusion.py`<br>`python scripts/run_lesson.py generative lesson_04_compact_latent_diffusion --dry-run` |
| L05 | `batch170-video-text-retrieval-lesson` | `.worktrees/batch170-video-text-retrieval-lesson` | `tracks/multimodal` | `video_text_retrieval_lesson` | `tracks/multimodal/lesson_17_video_text_retrieval/`, `tests/test_tracks_multimodal_video_text_retrieval.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_video_text_retrieval.py`<br>`python scripts/run_lesson.py multimodal lesson_17_video_text_retrieval --dry-run` |
| L06 | `batch170-prompt-learning-lesson` | `.worktrees/batch170-prompt-learning-lesson` | `tracks/multimodal` | `prompt_learning_lesson` | `tracks/multimodal/lesson_18_prompt_learning_vlm/`, `tests/test_tracks_multimodal_prompt_learning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_prompt_learning.py`<br>`python scripts/run_lesson.py multimodal lesson_18_prompt_learning_vlm --dry-run` |
| L07 | `batch170-crowd-counting-lesson` | `.worktrees/batch170-crowd-counting-lesson` | `tracks/vision` | `crowd_counting_lesson` | `tracks/vision/lesson_18_synthetic_crowd_counting/`, `tests/test_tracks_vision_crowd_counting.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_crowd_counting.py`<br>`python scripts/run_lesson.py vision lesson_18_synthetic_crowd_counting --dry-run` |
| L08 | `batch170-monocular-depth-lesson` | `.worktrees/batch170-monocular-depth-lesson` | `tracks/vision` | `monocular_depth_estimation_lesson` | `tracks/vision/lesson_19_synthetic_monocular_depth_estimation/`, `tests/test_tracks_vision_monocular_depth.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_monocular_depth.py`<br>`python scripts/run_lesson.py vision lesson_19_synthetic_monocular_depth_estimation --dry-run` |
| L09 | `batch170-text-matching-lesson` | `.worktrees/batch170-text-matching-lesson` | `tracks/nlp` | `text_matching_lesson` | `tracks/nlp/lesson_08_compact_text_matching_biencoder/`, `tests/test_tracks_nlp_text_matching.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_text_matching.py`<br>`python scripts/run_lesson.py nlp lesson_08_compact_text_matching_biencoder --dry-run` |
| L10 | `batch170-transformer-summarization-lesson` | `.worktrees/batch170-transformer-summarization-lesson` | `tracks/nlp` | `transformer_summarization_lesson` | `tracks/nlp/lesson_09_compact_transformer_summarization/`, `tests/test_tracks_nlp_transformer_summarization.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_transformer_summarization.py`<br>`python scripts/run_lesson.py nlp lesson_09_compact_transformer_summarization --dry-run` |

## Integration Branch Ownership

`batch170-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch170-chat-sft`
2. `batch170-mamba-llm`
3. `batch170-diffusion-lesson`
4. `batch170-latent-diffusion-lesson`
5. `batch170-video-text-retrieval-lesson`
6. `batch170-prompt-learning-lesson`
7. `batch170-crowd-counting-lesson`
8. `batch170-monocular-depth-lesson`
9. `batch170-text-matching-lesson`
10. `batch170-transformer-summarization-lesson`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
