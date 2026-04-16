# Batch 27 Track-Fill Manifest

Batch 27 keeps the fixed `10 worktree + 10 lane` loop moving immediately after the Batch 26 merge
back. This round leans into prompt-side deliberation for the LLM track, synthesis-oriented
generation, face-centric multimodal understanding, face-focused vision continuation, and
task-oriented NLP neighbor fills.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch27 face-and-dialog track-fill manifest`
- integration branch: `batch270-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch270-self-consistency-prompting` | `.worktrees/batch270-self-consistency-prompting` | `tracks/llm` | `self_consistency_prompting_llm_lesson` | `tracks/llm/lesson_22_toy_self_consistency_prompting/`, `tests/test_tracks_llm_self_consistency_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_self_consistency_prompting.py`<br>`python scripts/run_lesson.py llm lesson_22_toy_self_consistency_prompting --dry-run` |
| L02 | `batch270-critic-rerank-prompting` | `.worktrees/batch270-critic-rerank-prompting` | `tracks/llm` | `critic_rerank_prompting_llm_lesson` | `tracks/llm/lesson_23_toy_critic_rerank_prompting/`, `tests/test_tracks_llm_critic_rerank_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_critic_rerank_prompting.py`<br>`python scripts/run_lesson.py llm lesson_23_toy_critic_rerank_prompting --dry-run` |
| L03 | `batch270-multi-focus-fusion` | `.worktrees/batch270-multi-focus-fusion` | `tracks/generative` | `multi_focus_fusion_generative_lesson` | `tracks/generative/lesson_23_toy_diffusion_multi_focus_fusion/`, `tests/test_tracks_generative_multi_focus_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_multi_focus_fusion.py`<br>`python scripts/run_lesson.py generative lesson_23_toy_diffusion_multi_focus_fusion --dry-run` |
| L04 | `batch270-image-synthesis` | `.worktrees/batch270-image-synthesis` | `tracks/generative` | `image_synthesis_generative_lesson` | `tracks/generative/lesson_24_toy_diffusion_image_synthesis/`, `tests/test_tracks_generative_image_synthesis.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_image_synthesis.py`<br>`python scripts/run_lesson.py generative lesson_24_toy_diffusion_image_synthesis --dry-run` |
| L05 | `batch270-face-identity-vlm` | `.worktrees/batch270-face-identity-vlm` | `tracks/multimodal` | `face_identity_multimodal_lesson` | `tracks/multimodal/lesson_37_face_identity_vlm_recognition/`, `tests/test_tracks_multimodal_face_identity.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_identity.py`<br>`python scripts/run_lesson.py multimodal lesson_37_face_identity_vlm_recognition --dry-run` |
| L06 | `batch270-face-verification-vlm` | `.worktrees/batch270-face-verification-vlm` | `tracks/multimodal` | `face_verification_multimodal_lesson` | `tracks/multimodal/lesson_38_face_verification_vlm_reasoning/`, `tests/test_tracks_multimodal_face_verification.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_verification.py`<br>`python scripts/run_lesson.py multimodal lesson_38_face_verification_vlm_reasoning --dry-run` |
| L07 | `batch270-face-detection` | `.worktrees/batch270-face-detection` | `tracks/vision` | `face_detection_vision_lesson` | `tracks/vision/lesson_38_synthetic_face_detection/`, `tests/test_tracks_vision_face_detection.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_detection.py`<br>`python scripts/run_lesson.py vision lesson_38_synthetic_face_detection --dry-run` |
| L08 | `batch270-face-alignment` | `.worktrees/batch270-face-alignment` | `tracks/vision` | `face_alignment_vision_lesson` | `tracks/vision/lesson_39_synthetic_face_alignment/`, `tests/test_tracks_vision_face_alignment.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_alignment.py`<br>`python scripts/run_lesson.py vision lesson_39_synthetic_face_alignment --dry-run` |
| L09 | `batch270-semantic-textual-similarity` | `.worktrees/batch270-semantic-textual-similarity` | `tracks/nlp` | `semantic_textual_similarity_nlp_lesson` | `tracks/nlp/lesson_28_toy_semantic_textual_similarity/`, `tests/test_tracks_nlp_semantic_textual_similarity.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_semantic_textual_similarity.py`<br>`python scripts/run_lesson.py nlp lesson_28_toy_semantic_textual_similarity --dry-run` |
| L10 | `batch270-dialog-state-tracking` | `.worktrees/batch270-dialog-state-tracking` | `tracks/nlp` | `dialog_state_tracking_nlp_lesson` | `tracks/nlp/lesson_29_toy_dialog_state_tracking/`, `tests/test_tracks_nlp_dialog_state_tracking.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_state_tracking.py`<br>`python scripts/run_lesson.py nlp lesson_29_toy_dialog_state_tracking --dry-run` |

## Integration Branch Ownership

`batch270-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch270-self-consistency-prompting`
2. `batch270-critic-rerank-prompting`
3. `batch270-multi-focus-fusion`
4. `batch270-image-synthesis`
5. `batch270-face-identity-vlm`
6. `batch270-face-verification-vlm`
7. `batch270-face-detection`
8. `batch270-face-alignment`
9. `batch270-semantic-textual-similarity`
10. `batch270-dialog-state-tracking`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
