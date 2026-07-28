# Batch 30 Track-Fill Manifest

Batch 30 starts immediately after the Batch 29 merge-back and keeps the fixed `10 worktree + 10
lane` loop moving. This round extends the recent prompt-control arc on the LLM track, continues
reference/identity control on the generative track, pushes face-centric multimodal reasoning into
landmarks and parsing, broadens the face vision block into verification and identification, and
keeps the NLP track on compact dialog decision modeling.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch30 face-grounding dialog-control manifest`
- integration branch: `batch300-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch300-reference-grounded-prompting` | `.worktrees/batch300-reference-grounded-prompting` | `tracks/llm` | `reference_grounded_prompting_llm_lesson` | `tracks/llm/lesson_28_compact_reference_grounded_prompting/`, `tests/test_tracks_llm_reference_grounded_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_reference_grounded_prompting.py`<br>`python scripts/run_lesson.py llm lesson_28_compact_reference_grounded_prompting --dry-run` |
| L02 | `batch300-constraint-repair-prompting` | `.worktrees/batch300-constraint-repair-prompting` | `tracks/llm` | `constraint_repair_prompting_llm_lesson` | `tracks/llm/lesson_29_compact_constraint_repair_prompting/`, `tests/test_tracks_llm_constraint_repair_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_constraint_repair_prompting.py`<br>`python scripts/run_lesson.py llm lesson_29_compact_constraint_repair_prompting --dry-run` |
| L03 | `batch300-multi-reference-generation` | `.worktrees/batch300-multi-reference-generation` | `tracks/generative` | `multi_reference_generation_generative_lesson` | `tracks/generative/lesson_29_compact_diffusion_multi_reference_generation/`, `tests/test_tracks_generative_multi_reference_generation.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_multi_reference_generation.py`<br>`python scripts/run_lesson.py generative lesson_29_compact_diffusion_multi_reference_generation --dry-run` |
| L04 | `batch300-identity-preserving-editing` | `.worktrees/batch300-identity-preserving-editing` | `tracks/generative` | `identity_preserving_editing_generative_lesson` | `tracks/generative/lesson_30_compact_diffusion_identity_preserving_editing/`, `tests/test_tracks_generative_identity_preserving_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_identity_preserving_editing.py`<br>`python scripts/run_lesson.py generative lesson_30_compact_diffusion_identity_preserving_editing --dry-run` |
| L05 | `batch300-face-landmark-vlm` | `.worktrees/batch300-face-landmark-vlm` | `tracks/multimodal` | `face_landmark_multimodal_lesson` | `tracks/multimodal/lesson_43_face_landmark_vlm_reasoning/`, `tests/test_tracks_multimodal_face_landmark_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_landmark_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_43_face_landmark_vlm_reasoning --dry-run` |
| L06 | `batch300-face-parsing-vlm` | `.worktrees/batch300-face-parsing-vlm` | `tracks/multimodal` | `face_parsing_multimodal_lesson` | `tracks/multimodal/lesson_44_face_parsing_vlm_reasoning/`, `tests/test_tracks_multimodal_face_parsing_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_face_parsing_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_44_face_parsing_vlm_reasoning --dry-run` |
| L07 | `batch300-face-verification-vision` | `.worktrees/batch300-face-verification-vision` | `tracks/vision` | `face_verification_vision_lesson` | `tracks/vision/lesson_44_synthetic_face_verification/`, `tests/test_tracks_vision_face_verification.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_verification.py`<br>`python scripts/run_lesson.py vision lesson_44_synthetic_face_verification --dry-run` |
| L08 | `batch300-face-identification-vision` | `.worktrees/batch300-face-identification-vision` | `tracks/vision` | `face_identification_vision_lesson` | `tracks/vision/lesson_45_synthetic_face_identification/`, `tests/test_tracks_vision_face_identification.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_identification.py`<br>`python scripts/run_lesson.py vision lesson_45_synthetic_face_identification --dry-run` |
| L09 | `batch300-dialog-policy-prediction` | `.worktrees/batch300-dialog-policy-prediction` | `tracks/nlp` | `dialog_policy_prediction_nlp_lesson` | `tracks/nlp/lesson_34_compact_dialog_policy_prediction/`, `tests/test_tracks_nlp_dialog_policy_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_policy_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_34_compact_dialog_policy_prediction --dry-run` |
| L10 | `batch300-dialog-domain-prediction` | `.worktrees/batch300-dialog-domain-prediction` | `tracks/nlp` | `dialog_domain_prediction_nlp_lesson` | `tracks/nlp/lesson_35_compact_dialog_domain_prediction/`, `tests/test_tracks_nlp_dialog_domain_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_domain_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_35_compact_dialog_domain_prediction --dry-run` |

## Integration Branch Ownership

`batch300-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch300-reference-grounded-prompting`
2. `batch300-constraint-repair-prompting`
3. `batch300-multi-reference-generation`
4. `batch300-identity-preserving-editing`
5. `batch300-face-landmark-vlm`
6. `batch300-face-parsing-vlm`
7. `batch300-face-verification-vision`
8. `batch300-face-identification-vision`
9. `batch300-dialog-policy-prediction`
10. `batch300-dialog-domain-prediction`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
