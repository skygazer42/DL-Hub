# Batch 34 Track-Fill Manifest

Batch 34 starts immediately after the Batch 33 merge-back and keeps the fixed `10 worktree + 10
lane` loop moving. This round continues the LLM structured-output arc with EBNF and SQL
constraints, pushes the generative editing block into polygon-mask editing and layout-attribute
fusion, extends the multimodal face/body reasoning line into hand-pose and gesture queries, shifts
the vision continuation into finger-count and handedness recognition, and keeps the NLP track
inside compact support-ops timing and callback decision modeling.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- nearest teaching-neighbor fill after the current end of each track

Start point:

- base branch: `main`
- branch anchor commit: `docs: mark batch33 gaps merged`
- integration branch: `batch340-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch340-ebnf-constrained-prompting` | `.worktrees/batch340-ebnf-constrained-prompting` | `tracks/llm` | `ebnf_constrained_prompting_llm_lesson` | `tracks/llm/lesson_36_compact_ebnf_constrained_prompting/`, `tests/test_tracks_llm_ebnf_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_ebnf_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_36_compact_ebnf_constrained_prompting --dry-run` |
| L02 | `batch340-sql-constrained-prompting` | `.worktrees/batch340-sql-constrained-prompting` | `tracks/llm` | `sql_constrained_prompting_llm_lesson` | `tracks/llm/lesson_37_compact_sql_constrained_prompting/`, `tests/test_tracks_llm_sql_constrained_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_sql_constrained_prompting.py`<br>`python scripts/run_lesson.py llm lesson_37_compact_sql_constrained_prompting --dry-run` |
| L03 | `batch340-polygon-mask-editing` | `.worktrees/batch340-polygon-mask-editing` | `tracks/generative` | `polygon_mask_editing_generative_lesson` | `tracks/generative/lesson_37_compact_diffusion_polygon_mask_editing/`, `tests/test_tracks_generative_polygon_mask_editing.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_polygon_mask_editing.py`<br>`python scripts/run_lesson.py generative lesson_37_compact_diffusion_polygon_mask_editing --dry-run` |
| L04 | `batch340-layout-attribute-fusion` | `.worktrees/batch340-layout-attribute-fusion` | `tracks/generative` | `layout_attribute_fusion_generative_lesson` | `tracks/generative/lesson_38_compact_diffusion_layout_attribute_fusion/`, `tests/test_tracks_generative_layout_attribute_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_layout_attribute_fusion.py`<br>`python scripts/run_lesson.py generative lesson_38_compact_diffusion_layout_attribute_fusion --dry-run` |
| L05 | `batch340-hand-pose-vlm` | `.worktrees/batch340-hand-pose-vlm` | `tracks/multimodal` | `hand_pose_multimodal_lesson` | `tracks/multimodal/lesson_51_hand_pose_vlm_reasoning/`, `tests/test_tracks_multimodal_hand_pose_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_hand_pose_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_51_hand_pose_vlm_reasoning --dry-run` |
| L06 | `batch340-gesture-vlm` | `.worktrees/batch340-gesture-vlm` | `tracks/multimodal` | `gesture_multimodal_lesson` | `tracks/multimodal/lesson_52_gesture_vlm_reasoning/`, `tests/test_tracks_multimodal_gesture_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_gesture_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_52_gesture_vlm_reasoning --dry-run` |
| L07 | `batch340-finger-count-vision` | `.worktrees/batch340-finger-count-vision` | `tracks/vision` | `finger_count_estimation_vision_lesson` | `tracks/vision/lesson_52_synthetic_finger_count_estimation/`, `tests/test_tracks_vision_finger_count_estimation.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_finger_count_estimation.py`<br>`python scripts/run_lesson.py vision lesson_52_synthetic_finger_count_estimation --dry-run` |
| L08 | `batch340-handedness-vision` | `.worktrees/batch340-handedness-vision` | `tracks/vision` | `handedness_classification_vision_lesson` | `tracks/vision/lesson_53_synthetic_handedness_classification/`, `tests/test_tracks_vision_handedness_classification.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_handedness_classification.py`<br>`python scripts/run_lesson.py vision lesson_53_synthetic_handedness_classification --dry-run` |
| L09 | `batch340-dialog-resolution-time` | `.worktrees/batch340-dialog-resolution-time` | `tracks/nlp` | `dialog_resolution_time_prediction_nlp_lesson` | `tracks/nlp/lesson_42_compact_dialog_resolution_time_prediction/`, `tests/test_tracks_nlp_dialog_resolution_time_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_resolution_time_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_42_compact_dialog_resolution_time_prediction --dry-run` |
| L10 | `batch340-dialog-callback-prediction` | `.worktrees/batch340-dialog-callback-prediction` | `tracks/nlp` | `dialog_callback_prediction_nlp_lesson` | `tracks/nlp/lesson_43_compact_dialog_callback_prediction/`, `tests/test_tracks_nlp_dialog_callback_prediction.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_dialog_callback_prediction.py`<br>`python scripts/run_lesson.py nlp lesson_43_compact_dialog_callback_prediction --dry-run` |

## Integration Branch Ownership

`batch340-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch340-ebnf-constrained-prompting`
2. `batch340-sql-constrained-prompting`
3. `batch340-polygon-mask-editing`
4. `batch340-layout-attribute-fusion`
5. `batch340-hand-pose-vlm`
6. `batch340-gesture-vlm`
7. `batch340-finger-count-vision`
8. `batch340-handedness-vision`
9. `batch340-dialog-resolution-time`
10. `batch340-dialog-callback-prediction`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
