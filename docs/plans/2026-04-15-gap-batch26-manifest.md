# Batch 26 Track-Fill Manifest

Batch 26 keeps the fixed `10 worktree + 10 lane` loop moving immediately after the Batch 25 merge
back. This round extends prompt-centric LLM agents, diffusion-driven image transformation, face and
text understanding, and classic NLP structure prediction while preserving the same balanced
track-first cadence.

Selection rules locked for this batch:

- fixed `10 worktree + 10 lane` execution
- track-first coverage across `tracks/llm`, `tracks/generative`, `tracks/multimodal`,
  `tracks/vision`, and `tracks/nlp`
- max 2 lanes per surface
- remaining direct user-tag mappings first, then nearest teaching-neighbor fill for the same track

Start point:

- base branch: `main`
- branch anchor commit: `plan: seed batch26 track-fill manifest`
- integration branch: `batch260-integration`
- worktree root: `.worktrees/`

## Lane Table

| lane_id | branch | worktree | surface | slug | lane_owned_paths | integration_owned_paths | verification_cmds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| L01 | `batch260-react-tool-prompting` | `.worktrees/batch260-react-tool-prompting` | `tracks/llm` | `react_tool_prompting_llm_lesson` | `tracks/llm/lesson_20_compact_react_tool_prompting/`, `tests/test_tracks_llm_react_tool_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_react_tool_prompting.py`<br>`python scripts/run_lesson.py llm lesson_20_compact_react_tool_prompting --dry-run` |
| L02 | `batch260-tree-of-thought-prompting` | `.worktrees/batch260-tree-of-thought-prompting` | `tracks/llm` | `tree_of_thought_prompting_llm_lesson` | `tracks/llm/lesson_21_compact_tree_of_thought_prompting/`, `tests/test_tracks_llm_tree_of_thought_prompting.py` | `README.md`, `tracks/llm/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_llm_tree_of_thought_prompting.py`<br>`python scripts/run_lesson.py llm lesson_21_compact_tree_of_thought_prompting --dry-run` |
| L03 | `batch260-diffusion-image-fusion` | `.worktrees/batch260-diffusion-image-fusion` | `tracks/generative` | `diffusion_image_fusion_generative_lesson` | `tracks/generative/lesson_21_compact_diffusion_image_fusion/`, `tests/test_tracks_generative_diffusion_image_fusion.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_image_fusion.py`<br>`python scripts/run_lesson.py generative lesson_21_compact_diffusion_image_fusion --dry-run` |
| L04 | `batch260-diffusion-style-transfer` | `.worktrees/batch260-diffusion-style-transfer` | `tracks/generative` | `diffusion_style_transfer_generative_lesson` | `tracks/generative/lesson_22_compact_diffusion_style_transfer/`, `tests/test_tracks_generative_diffusion_style_transfer.py` | `README.md`, `tracks/generative/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_generative_diffusion_style_transfer.py`<br>`python scripts/run_lesson.py generative lesson_22_compact_diffusion_style_transfer --dry-run` |
| L05 | `batch260-facial-expression-vlm` | `.worktrees/batch260-facial-expression-vlm` | `tracks/multimodal` | `facial_expression_multimodal_lesson` | `tracks/multimodal/lesson_35_face_expression_vlm_recognition/`, `tests/test_tracks_multimodal_facial_expression.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_facial_expression.py`<br>`python scripts/run_lesson.py multimodal lesson_35_face_expression_vlm_recognition --dry-run` |
| L06 | `batch260-deepfake-reasoning` | `.worktrees/batch260-deepfake-reasoning` | `tracks/multimodal` | `deepfake_reasoning_multimodal_lesson` | `tracks/multimodal/lesson_36_face_anti_spoof_vlm_reasoning/`, `tests/test_tracks_multimodal_deepfake_reasoning.py` | `README.md`, `tracks/multimodal/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_multimodal_deepfake_reasoning.py`<br>`python scripts/run_lesson.py multimodal lesson_36_face_anti_spoof_vlm_reasoning --dry-run` |
| L07 | `batch260-text-recognition` | `.worktrees/batch260-text-recognition` | `tracks/vision` | `text_recognition_vision_lesson` | `tracks/vision/lesson_36_synthetic_text_recognition/`, `tests/test_tracks_vision_text_recognition.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_text_recognition.py`<br>`python scripts/run_lesson.py vision lesson_36_synthetic_text_recognition --dry-run` |
| L08 | `batch260-face-parsing` | `.worktrees/batch260-face-parsing` | `tracks/vision` | `face_parsing_vision_lesson` | `tracks/vision/lesson_37_synthetic_face_parsing/`, `tests/test_tracks_vision_face_parsing.py` | `README.md`, `tracks/vision/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_vision_face_parsing.py`<br>`python scripts/run_lesson.py vision lesson_37_synthetic_face_parsing --dry-run` |
| L09 | `batch260-joint-intent-slot` | `.worktrees/batch260-joint-intent-slot` | `tracks/nlp` | `joint_intent_slot_nlp_lesson` | `tracks/nlp/lesson_26_compact_joint_intent_slot_parsing/`, `tests/test_tracks_nlp_joint_intent_slot.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_joint_intent_slot.py`<br>`python scripts/run_lesson.py nlp lesson_26_compact_joint_intent_slot_parsing --dry-run` |
| L10 | `batch260-textual-entailment` | `.worktrees/batch260-textual-entailment` | `tracks/nlp` | `textual_entailment_nlp_lesson` | `tracks/nlp/lesson_27_compact_textual_entailment/`, `tests/test_tracks_nlp_textual_entailment.py` | `README.md`, `tracks/nlp/README.md`, `tests/test_scripts_run_lesson.py` | `pytest -q tests/test_tracks_nlp_textual_entailment.py`<br>`python scripts/run_lesson.py nlp lesson_27_compact_textual_entailment --dry-run` |

## Integration Branch Ownership

`batch260-integration` owns only the shared surfaces required after the 10 lane merges:

- `README.md`
- `tracks/llm/README.md`
- `tracks/generative/README.md`
- `tracks/multimodal/README.md`
- `tracks/vision/README.md`
- `tracks/nlp/README.md`
- `tests/test_scripts_run_lesson.py`
- any shared verification-only updates required to make the merged track batch pass on `main`

## Merge Order

1. `batch260-react-tool-prompting`
2. `batch260-tree-of-thought-prompting`
3. `batch260-diffusion-image-fusion`
4. `batch260-diffusion-style-transfer`
5. `batch260-facial-expression-vlm`
6. `batch260-deepfake-reasoning`
7. `batch260-text-recognition`
8. `batch260-face-parsing`
9. `batch260-joint-intent-slot`
10. `batch260-textual-entailment`

## Exit Criteria

- each lane has at least one local commit on its own branch
- each lane passes its manifest smoke commands from its own worktree
- integration updates the top-level lesson counts and track lesson tables for all 10 new lessons
- integration branch passes fresh shared verification before merge back to `main`
