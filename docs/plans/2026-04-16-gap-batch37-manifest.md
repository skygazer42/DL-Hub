# Batch 37 Track-Fill Manifest

Batch 37 starts immediately after the Batch 36 merge-back and keeps the fixed `10 worktree +
10 lane` execution loop. This round extends the LLM structured-output block into INI and TSV,
pushes the generative editing/fusion arc into path-mask editing and layout-lighting fusion,
continues multimodal hand understanding with finger-spread and thumb-position reasoning, advances
the vision hand track into finger-curvature and thumb-contact estimation, and keeps the NLP
support-ops line moving with resolution-action and owner-handoff prediction.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch36 gaps merged`
- integration branch: `batch370-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch370-ini-constrained-prompting` | `.worktrees/batch370-ini-constrained-prompting` | `tracks/llm` | `ini_constrained_prompting_llm_lesson` | `tracks/llm/lesson_42_toy_ini_constrained_prompting/`, `tests/test_tracks_llm_ini_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_ini_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_42_toy_ini_constrained_prompting --dry-run` |
| L02 | `batch370-tsv-constrained-prompting` | `.worktrees/batch370-tsv-constrained-prompting` | `tracks/llm` | `tsv_constrained_prompting_llm_lesson` | `tracks/llm/lesson_43_toy_tsv_constrained_prompting/`, `tests/test_tracks_llm_tsv_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_tsv_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_43_toy_tsv_constrained_prompting --dry-run` |
| L03 | `batch370-path-mask-editing` | `.worktrees/batch370-path-mask-editing` | `tracks/generative` | `path_mask_editing_generative_lesson` | `tracks/generative/lesson_43_toy_diffusion_path_mask_editing/`, `tests/test_tracks_generative_path_mask_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_path_mask_editing.py`<br>`python scripts/run_lesson.py generative lesson_43_toy_diffusion_path_mask_editing --dry-run` |
| L04 | `batch370-layout-lighting-fusion` | `.worktrees/batch370-layout-lighting-fusion` | `tracks/generative` | `layout_lighting_fusion_generative_lesson` | `tracks/generative/lesson_44_toy_diffusion_layout_lighting_fusion/`, `tests/test_tracks_generative_layout_lighting_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_lighting_fusion.py`<br>`python scripts/run_lesson.py generative lesson_44_toy_diffusion_layout_lighting_fusion --dry-run` |
| L05 | `batch370-finger-spread-vlm` | `.worktrees/batch370-finger-spread-vlm` | `tracks/multimodal` | `finger_spread_multimodal_lesson` | `tracks/multimodal/lesson_57_finger_spread_vlm_reasoning/`, `tests/test_tracks_multimodal_finger_spread_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_finger_spread_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_57_finger_spread_vlm_reasoning --dry-run` |
| L06 | `batch370-thumb-position-vlm` | `.worktrees/batch370-thumb-position-vlm` | `tracks/multimodal` | `thumb_position_multimodal_lesson` | `tracks/multimodal/lesson_58_thumb_position_vlm_reasoning/`, `tests/test_tracks_multimodal_thumb_position_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_thumb_position_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_58_thumb_position_vlm_reasoning --dry-run` |
| L07 | `batch370-finger-curvature-vision` | `.worktrees/batch370-finger-curvature-vision` | `tracks/vision` | `finger_curvature_estimation_vision_lesson` | `tracks/vision/lesson_58_synthetic_finger_curvature_estimation/`, `tests/test_tracks_vision_finger_curvature_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_finger_curvature_estimation.py`<br>`python scripts/run_lesson.py vision lesson_58_synthetic_finger_curvature_estimation --dry-run` |
| L08 | `batch370-thumb-contact-vision` | `.worktrees/batch370-thumb-contact-vision` | `tracks/vision` | `thumb_contact_classification_vision_lesson` | `tracks/vision/lesson_59_synthetic_thumb_contact_classification/`, `tests/test_tracks_vision_thumb_contact_classification.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_thumb_contact_classification.py`<br>`python scripts/run_lesson.py vision lesson_59_synthetic_thumb_contact_classification --dry-run` |
| L09 | `batch370-dialog-resolution-action` | `.worktrees/batch370-dialog-resolution-action` | `tracks/nlp` | `dialog_resolution_action_prediction_nlp_lesson` | `tracks/nlp/lesson_48_toy_dialog_resolution_action_prediction/`, `tests/test_tracks_nlp_dialog_resolution_action_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_resolution_action_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_48_toy_dialog_resolution_action_prediction --dry-run` |
| L10 | `batch370-dialog-owner-handoff` | `.worktrees/batch370-dialog-owner-handoff` | `tracks/nlp` | `dialog_owner_handoff_prediction_nlp_lesson` | `tracks/nlp/lesson_49_toy_dialog_owner_handoff_prediction/`, `tests/test_tracks_nlp_dialog_owner_handoff_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_owner_handoff_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_49_toy_dialog_owner_handoff_prediction --dry-run` |

## Integration Branch Ownership

`batch370-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch370-ini-constrained-prompting`
2. `batch370-tsv-constrained-prompting`
3. `batch370-path-mask-editing`
4. `batch370-layout-lighting-fusion`
5. `batch370-finger-spread-vlm`
6. `batch370-thumb-position-vlm`
7. `batch370-finger-curvature-vision`
8. `batch370-thumb-contact-vision`
9. `batch370-dialog-resolution-action`
10. `batch370-dialog-owner-handoff`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
