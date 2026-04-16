import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_run_lesson_lists_tracks() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Tracks:" in proc.stdout
    assert "multimodal" in proc.stdout
    assert "vision" in proc.stdout


def test_run_lesson_lists_lessons_for_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "vision", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (vision):" in proc.stdout
    assert "lesson_01_mnist_lenet" in proc.stdout
    assert "lesson_14_video_mot_basics" in proc.stdout
    assert "lesson_18_synthetic_crowd_counting" in proc.stdout
    assert "lesson_19_synthetic_monocular_depth_estimation" in proc.stdout
    assert "lesson_20_synthetic_lane_detection" in proc.stdout
    assert "lesson_21_synthetic_lane_topology_estimation" in proc.stdout
    assert "lesson_22_synthetic_road_scene_understanding" in proc.stdout
    assert "lesson_23_synthetic_image_dehazing" in proc.stdout
    assert "lesson_24_synthetic_reflection_removal" in proc.stdout
    assert "lesson_25_synthetic_image_fusion" in proc.stdout
    assert "lesson_26_synthetic_text_detection" in proc.stdout
    assert "lesson_27_synthetic_edge_detection" in proc.stdout
    assert "lesson_28_synthetic_salient_object_detection" in proc.stdout
    assert "lesson_29_synthetic_camouflaged_object_detection" in proc.stdout
    assert "lesson_30_synthetic_salient_object_detection_boxes" in proc.stdout
    assert "lesson_31_synthetic_interactive_segmentation" in proc.stdout
    assert "lesson_32_synthetic_face_landmark_detection" in proc.stdout
    assert "lesson_33_synthetic_face_liveness_detection" in proc.stdout
    assert "lesson_34_synthetic_license_plate_recognition" in proc.stdout
    assert "lesson_35_synthetic_6d_pose_estimation" in proc.stdout
    assert "lesson_36_synthetic_text_recognition" in proc.stdout
    assert "lesson_37_synthetic_face_parsing" in proc.stdout
    assert "lesson_38_synthetic_face_detection" in proc.stdout
    assert "lesson_39_synthetic_face_alignment" in proc.stdout
    assert "lesson_40_synthetic_face_attribute_recognition" in proc.stdout
    assert "lesson_41_synthetic_face_occlusion_estimation" in proc.stdout
    assert "lesson_42_synthetic_face_expression_recognition" in proc.stdout
    assert "lesson_43_synthetic_deepfake_detection" in proc.stdout
    assert "lesson_44_synthetic_face_verification" in proc.stdout
    assert "lesson_45_synthetic_face_identification" in proc.stdout
    assert "lesson_46_synthetic_face_retrieval" in proc.stdout
    assert "lesson_47_synthetic_face_pose_estimation" in proc.stdout
    assert "lesson_48_synthetic_gaze_estimation" in proc.stdout
    assert "lesson_49_synthetic_human_pose_estimation" in proc.stdout
    assert "lesson_50_synthetic_hand_pose_estimation" in proc.stdout
    assert "lesson_51_synthetic_gesture_recognition" in proc.stdout
    assert "lesson_52_synthetic_finger_count_estimation" in proc.stdout
    assert "lesson_53_synthetic_handedness_classification" in proc.stdout
    assert "lesson_54_synthetic_palm_orientation_estimation" in proc.stdout
    assert "lesson_55_synthetic_sign_digit_classification" in proc.stdout
    assert "lesson_56_synthetic_finger_spread_estimation" in proc.stdout
    assert "lesson_57_synthetic_thumb_position_classification" in proc.stdout
    assert "lesson_58_synthetic_finger_curvature_estimation" in proc.stdout
    assert "lesson_59_synthetic_thumb_contact_classification" in proc.stdout
    assert "lesson_60_synthetic_image_deraining" in proc.stdout
    assert "lesson_61_synthetic_image_retrieval" in proc.stdout
    assert "lesson_62_synthetic_image_matching" in proc.stdout
    assert "lesson_63_synthetic_image_stitching" in proc.stdout
    assert "lesson_64_synthetic_fine_grained_recognition" in proc.stdout
    assert "lesson_65_synthetic_few_shot_recognition" in proc.stdout
    assert "lesson_66_synthetic_video_object_detection" in proc.stdout
    assert "lesson_67_synthetic_video_stabilization" in proc.stdout
    assert "lesson_68_synthetic_video_frame_interpolation" in proc.stdout
    assert "lesson_69_synthetic_video_restoration" in proc.stdout
    assert "lesson_70_synthetic_video_understanding" in proc.stdout
    assert "lesson_71_synthetic_video_summarization" in proc.stdout
    assert "lesson_72_synthetic_video_enhancement" in proc.stdout
    assert "lesson_73_synthetic_video_object_segmentation" in proc.stdout
    assert "lesson_74_synthetic_video_instance_segmentation" in proc.stdout
    assert "lesson_75_synthetic_video_matting" in proc.stdout
    assert "lesson_76_synthetic_image_deweathering" in proc.stdout
    assert "lesson_77_synthetic_transparent_depth_estimation" in proc.stdout
    assert "lesson_78_synthetic_image_relighting" in proc.stdout
    assert "lesson_79_synthetic_transparent_object_segmentation" in proc.stdout
    assert "lesson_80_synthetic_event_camera_understanding" in proc.stdout
    assert "lesson_81_synthetic_shadow_detection" in proc.stdout
    assert "lesson_82_synthetic_layout_generation" in proc.stdout
    assert "lesson_83_synthetic_panoptic_segmentation" in proc.stdout
    assert "lesson_84_synthetic_medical_segmentation" in proc.stdout
    assert "lesson_85_synthetic_scene_text_spotting" in proc.stdout
    assert "lesson_86_synthetic_co_segmentation" in proc.stdout
    assert "lesson_87_synthetic_action_recognition" in proc.stdout
    assert "lesson_88_synthetic_reid" in proc.stdout
    assert "lesson_89_synthetic_anomaly_detection" in proc.stdout


def test_run_lesson_dry_run_resolves_train_module() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "vision", "lesson_01_mnist_lenet", "--dry-run"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_01_mnist_lenet.train" in proc.stdout


def test_run_lesson_lists_lessons_for_multimodal_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "multimodal", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (multimodal):" in proc.stdout
    assert "lesson_01_clip_toy_retrieval" in proc.stdout
    assert "lesson_02_blip_toy_captioning" in proc.stdout
    assert "lesson_03_llava_toy_instruction_vlm" in proc.stdout
    assert "lesson_04_grounding_toy_refexp" in proc.stdout
    assert "lesson_05_mask_grounding_toy_refexp" in proc.stdout
    assert "lesson_06_flamingo_toy_interleaved_vlm" in proc.stdout
    assert "lesson_07_qformer_toy_bridge_vlm" in proc.stdout
    assert "lesson_08_perceiver_resampler_toy_vlm" in proc.stdout
    assert "lesson_09_paligemma_toy_siglip_decoder_vlm" in proc.stdout
    assert "lesson_10_owlvit_toy_open_vocab_detection" in proc.stdout
    assert "lesson_11_grounded_sam_toy_open_vocab_segmentation" in proc.stdout
    assert "lesson_12_key_value_ocr_toy_doc_vlm" in proc.stdout
    assert "lesson_13_video_vlm_toy_temporal_qa" in proc.stdout
    assert "lesson_14_bmn_toy_temporal_grounding" in proc.stdout
    assert "lesson_15_2dtan_toy_temporal_grounding" in proc.stdout
    assert "lesson_16_multiscale_2dtan_toy_temporal_grounding" in proc.stdout
    assert "lesson_17_video_text_retrieval" in proc.stdout
    assert "lesson_18_prompt_learning_vlm" in proc.stdout
    assert "lesson_19_audio_text_understanding" in proc.stdout
    assert "lesson_20_audio_visual_learning" in proc.stdout
    assert "lesson_21_audio_grounded_retrieval" in proc.stdout
    assert "lesson_22_audio_visual_event_localization" in proc.stdout
    assert "lesson_23_embodied_question_answering" in proc.stdout
    assert "lesson_24_multimodal_reasoning" in proc.stdout
    assert "lesson_25_vision_language_navigation" in proc.stdout
    assert "lesson_26_image_text_reranking" in proc.stdout
    assert "lesson_27_scene_text_vlm_recognition" in proc.stdout
    assert "lesson_28_document_vlm_reasoning" in proc.stdout
    assert "lesson_29_human_object_interaction_reasoning" in proc.stdout
    assert "lesson_30_vision_language_gaze_estimation" in proc.stdout
    assert "lesson_31_person_search_attribute_retrieval" in proc.stdout
    assert "lesson_32_video_text_action_localization" in proc.stdout
    assert "lesson_33_pedestrian_attribute_recognition" in proc.stdout
    assert "lesson_34_video_text_action_recognition" in proc.stdout
    assert "lesson_35_face_expression_vlm_recognition" in proc.stdout
    assert "lesson_36_face_anti_spoof_vlm_reasoning" in proc.stdout
    assert "lesson_37_face_identity_vlm_recognition" in proc.stdout
    assert "lesson_38_face_verification_vlm_reasoning" in proc.stdout
    assert "lesson_39_face_attribute_vlm_reasoning" in proc.stdout
    assert "lesson_40_face_caption_vlm_grounding" in proc.stdout
    assert "lesson_41_face_occlusion_vlm_reasoning" in proc.stdout
    assert "lesson_42_face_region_grounding_vlm" in proc.stdout
    assert "lesson_43_face_landmark_vlm_reasoning" in proc.stdout
    assert "lesson_44_face_parsing_vlm_reasoning" in proc.stdout
    assert "lesson_45_face_alignment_vlm_reasoning" in proc.stdout
    assert "lesson_46_face_detection_vlm_reasoning" in proc.stdout
    assert "lesson_47_face_retrieval_vlm_reasoning" in proc.stdout
    assert "lesson_48_face_pose_vlm_reasoning" in proc.stdout
    assert "lesson_49_face_gaze_vlm_reasoning" in proc.stdout
    assert "lesson_50_person_pose_vlm_reasoning" in proc.stdout
    assert "lesson_51_hand_pose_vlm_reasoning" in proc.stdout
    assert "lesson_52_gesture_vlm_reasoning" in proc.stdout
    assert "lesson_53_finger_count_vlm_reasoning" in proc.stdout
    assert "lesson_54_handedness_vlm_reasoning" in proc.stdout
    assert "lesson_55_palm_orientation_vlm_reasoning" in proc.stdout
    assert "lesson_56_sign_digit_vlm_reasoning" in proc.stdout
    assert "lesson_57_finger_spread_vlm_reasoning" in proc.stdout
    assert "lesson_58_thumb_position_vlm_reasoning" in proc.stdout


def test_run_lesson_lists_lessons_for_llm_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "llm", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (llm):" in proc.stdout
    assert "lesson_01_toy_causal_lm_transformer" in proc.stdout
    assert "lesson_02_toy_chat_sft" in proc.stdout
    assert "lesson_03_toy_mamba_language_model" in proc.stdout
    assert "lesson_04_toy_instruction_tuning" in proc.stdout
    assert "lesson_05_toy_prefix_tuning" in proc.stdout
    assert "lesson_06_toy_preference_optimization" in proc.stdout
    assert "lesson_07_toy_reward_modeling" in proc.stdout
    assert "lesson_08_toy_span_corruption" in proc.stdout
    assert "lesson_09_toy_rlhf_ppo" in proc.stdout
    assert "lesson_10_toy_grpo_alignment" in proc.stdout
    assert "lesson_11_toy_rag_language_model" in proc.stdout
    assert "lesson_12_toy_transformer_interpretability" in proc.stdout
    assert "lesson_13_toy_tool_calling_agent" in proc.stdout
    assert "lesson_14_toy_replaced_token_detection_transformer" in proc.stdout
    assert "lesson_15_toy_llm_judge" in proc.stdout
    assert "lesson_16_toy_multi_turn_memory_sft" in proc.stdout
    assert "lesson_17_toy_self_refine_prompting" in proc.stdout
    assert "lesson_18_toy_reflection_memory_agent" in proc.stdout
    assert "lesson_19_toy_plan_execute_prompting" in proc.stdout
    assert "lesson_20_toy_react_tool_prompting" in proc.stdout
    assert "lesson_21_toy_tree_of_thought_prompting" in proc.stdout
    assert "lesson_22_toy_self_consistency_prompting" in proc.stdout
    assert "lesson_23_toy_critic_rerank_prompting" in proc.stdout
    assert "lesson_24_toy_debate_prompting" in proc.stdout
    assert "lesson_25_toy_verifier_guided_prompting" in proc.stdout
    assert "lesson_26_toy_process_supervision_prompting" in proc.stdout
    assert "lesson_27_toy_self_correction_prompting" in proc.stdout
    assert "lesson_28_toy_reference_grounded_prompting" in proc.stdout
    assert "lesson_29_toy_constraint_repair_prompting" in proc.stdout
    assert "lesson_30_toy_citation_grounded_prompting" in proc.stdout
    assert "lesson_31_toy_schema_constrained_prompting" in proc.stdout
    assert "lesson_32_toy_json_constrained_prompting" in proc.stdout
    assert "lesson_33_toy_function_signature_prompting" in proc.stdout
    assert "lesson_34_toy_xml_constrained_prompting" in proc.stdout
    assert "lesson_35_toy_regex_constrained_prompting" in proc.stdout
    assert "lesson_36_toy_ebnf_constrained_prompting" in proc.stdout
    assert "lesson_37_toy_sql_constrained_prompting" in proc.stdout
    assert "lesson_38_toy_yaml_constrained_prompting" in proc.stdout
    assert "lesson_39_toy_csv_constrained_prompting" in proc.stdout
    assert "lesson_40_toy_toml_constrained_prompting" in proc.stdout
    assert "lesson_41_toy_markdown_table_constrained_prompting" in proc.stdout
    assert "lesson_42_toy_ini_constrained_prompting" in proc.stdout
    assert "lesson_43_toy_tsv_constrained_prompting" in proc.stdout


def test_run_lesson_lists_lessons_for_generative_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "generative", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (generative):" in proc.stdout
    assert "lesson_01_vae_mnist" in proc.stdout
    assert "lesson_02_gan_mnist" in proc.stdout
    assert "lesson_03_toy_diffusion_mnist" in proc.stdout
    assert "lesson_04_toy_latent_diffusion" in proc.stdout
    assert "lesson_05_toy_consistency_model" in proc.stdout
    assert "lesson_06_toy_flow_matching" in proc.stdout
    assert "lesson_07_toy_rectified_flow" in proc.stdout
    assert "lesson_08_toy_diffusion_transformer" in proc.stdout
    assert "lesson_09_toy_conditional_gan" in proc.stdout
    assert "lesson_10_toy_diffusion_image_editing" in proc.stdout
    assert "lesson_11_toy_controlnet" in proc.stdout
    assert "lesson_12_toy_layout_to_image" in proc.stdout
    assert "lesson_13_toy_text_to_image_diffusion" in proc.stdout
    assert "lesson_14_toy_diffusion_inpainting" in proc.stdout
    assert "lesson_15_toy_diffusion_super_resolution" in proc.stdout
    assert "lesson_16_toy_diffusion_deblurring" in proc.stdout
    assert "lesson_17_toy_diffusion_denoising" in proc.stdout
    assert "lesson_18_toy_diffusion_deraining" in proc.stdout
    assert "lesson_19_toy_diffusion_dehazing" in proc.stdout
    assert "lesson_20_toy_diffusion_reflection_removal" in proc.stdout
    assert "lesson_21_toy_diffusion_image_fusion" in proc.stdout
    assert "lesson_22_toy_diffusion_style_transfer" in proc.stdout
    assert "lesson_23_toy_diffusion_multi_focus_fusion" in proc.stdout
    assert "lesson_24_toy_diffusion_image_synthesis" in proc.stdout
    assert "lesson_25_toy_diffusion_compositional_generation" in proc.stdout
    assert "lesson_26_toy_diffusion_image_variation" in proc.stdout
    assert "lesson_27_toy_diffusion_reference_guided_generation" in proc.stdout
    assert "lesson_28_toy_diffusion_subject_driven_generation" in proc.stdout
    assert "lesson_29_toy_diffusion_multi_reference_generation" in proc.stdout
    assert "lesson_30_toy_diffusion_identity_preserving_editing" in proc.stdout
    assert "lesson_31_toy_diffusion_reference_editing" in proc.stdout
    assert "lesson_32_toy_diffusion_layout_preserving_editing" in proc.stdout
    assert "lesson_33_toy_diffusion_masked_reference_editing" in proc.stdout
    assert "lesson_34_toy_diffusion_layout_reference_fusion" in proc.stdout
    assert "lesson_35_toy_diffusion_box_mask_editing" in proc.stdout
    assert "lesson_36_toy_diffusion_layout_subject_fusion" in proc.stdout
    assert "lesson_37_toy_diffusion_polygon_mask_editing" in proc.stdout
    assert "lesson_38_toy_diffusion_layout_attribute_fusion" in proc.stdout
    assert "lesson_39_toy_diffusion_scribble_mask_editing" in proc.stdout
    assert "lesson_40_toy_diffusion_layout_style_fusion" in proc.stdout
    assert "lesson_41_toy_diffusion_stroke_mask_editing" in proc.stdout
    assert "lesson_42_toy_diffusion_layout_palette_fusion" in proc.stdout
    assert "lesson_43_toy_diffusion_path_mask_editing" in proc.stdout
    assert "lesson_44_toy_diffusion_layout_lighting_fusion" in proc.stdout
    assert "lesson_45_toy_video_diffusion" in proc.stdout
    assert "lesson_46_toy_image_to_video_diffusion" in proc.stdout
    assert "lesson_47_toy_text_to_3d" in proc.stdout
    assert "lesson_48_toy_image_to_3d" in proc.stdout
    assert "lesson_49_toy_text_to_video" in proc.stdout
    assert "lesson_50_toy_video_to_video" in proc.stdout
    assert "lesson_51_toy_world_models" in proc.stdout


def test_run_lesson_lists_lessons_for_pointcloud_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "pointcloud", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (pointcloud):" in proc.stdout
    assert "lesson_01_pointnet_toy_classification" in proc.stdout
    assert "lesson_07_pointnet_toy_reconstruction" in proc.stdout
    assert "lesson_23_pointcloud_selfsupervised_ressl" in proc.stdout
    assert "lesson_24_toy_pointcloud_completion" in proc.stdout
    assert "lesson_25_toy_scene_flow_estimation" in proc.stdout
    assert "lesson_26_toy_gaussian_splatting" in proc.stdout
    assert "lesson_27_toy_3d_object_detection" in proc.stdout
    assert "lesson_28_toy_3d_semantic_segmentation" in proc.stdout
    assert "lesson_29_toy_3d_instance_segmentation" in proc.stdout
    assert "lesson_30_toy_3d_object_tracking" in proc.stdout
    assert "lesson_31_toy_open_vocabulary_3d" in proc.stdout
    assert "lesson_32_toy_pointcloud_forecasting" in proc.stdout
    assert "lesson_33_toy_pointcloud_anomaly_detection" in proc.stdout
    assert "lesson_34_toy_pointcloud_upsampling" in proc.stdout
    assert "lesson_35_toy_shape_correspondence_3d" in proc.stdout
    assert "lesson_36_toy_pointcloud_registration" in proc.stdout


def test_run_lesson_lists_lessons_for_nlp_track() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", "nlp", "--list"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "Lessons (nlp):" in proc.stdout
    assert "lesson_01_toy_text_classification" in proc.stdout
    assert "lesson_07_reading_comprehension" in proc.stdout
    assert "lesson_08_toy_text_matching_biencoder" in proc.stdout
    assert "lesson_09_toy_transformer_summarization" in proc.stdout
    assert "lesson_10_toy_prompt_tuning_classifier" in proc.stdout
    assert "lesson_11_toy_few_shot_text_classification" in proc.stdout
    assert "lesson_12_toy_in_context_text_classification" in proc.stdout
    assert "lesson_13_toy_masked_language_modeling" in proc.stdout
    assert "lesson_14_toy_contrastive_sentence_embedding" in proc.stdout
    assert "lesson_15_toy_cross_encoder_reranking" in proc.stdout
    assert "lesson_16_toy_text_clustering" in proc.stdout
    assert "lesson_17_toy_text_anomaly_detection" in proc.stdout
    assert "lesson_18_toy_topic_modeling" in proc.stdout
    assert "lesson_19_toy_distilled_text_classifier" in proc.stdout
    assert "lesson_20_toy_adversarial_text_classification" in proc.stdout
    assert "lesson_21_toy_adversarial_example_detection" in proc.stdout
    assert "lesson_22_toy_weak_supervision_text_classification" in proc.stdout
    assert "lesson_23_toy_sentence_denoising_autoencoder" in proc.stdout
    assert "lesson_24_toy_meta_few_shot_text_classification" in proc.stdout
    assert "lesson_25_toy_low_shot_intent_detection" in proc.stdout
    assert "lesson_26_toy_joint_intent_slot_parsing" in proc.stdout
    assert "lesson_27_toy_textual_entailment" in proc.stdout
    assert "lesson_28_toy_semantic_textual_similarity" in proc.stdout
    assert "lesson_29_toy_dialog_state_tracking" in proc.stdout
    assert "lesson_30_toy_dialog_response_selection" in proc.stdout
    assert "lesson_31_toy_slot_carryover_prediction" in proc.stdout
    assert "lesson_32_toy_dialog_act_prediction" in proc.stdout
    assert "lesson_33_toy_dialog_intent_prediction" in proc.stdout
    assert "lesson_34_toy_dialog_policy_prediction" in proc.stdout
    assert "lesson_35_toy_dialog_domain_prediction" in proc.stdout
    assert "lesson_36_toy_dialog_slot_prediction" in proc.stdout
    assert "lesson_37_toy_dialog_outcome_prediction" in proc.stdout
    assert "lesson_38_toy_dialog_satisfaction_prediction" in proc.stdout
    assert "lesson_39_toy_dialog_escalation_risk_prediction" in proc.stdout
    assert "lesson_40_toy_dialog_priority_prediction" in proc.stdout
    assert "lesson_41_toy_dialog_transfer_prediction" in proc.stdout
    assert "lesson_42_toy_dialog_resolution_time_prediction" in proc.stdout
    assert "lesson_43_toy_dialog_callback_prediction" in proc.stdout
    assert "lesson_44_toy_dialog_sla_breach_prediction" in proc.stdout
    assert "lesson_45_toy_dialog_followup_channel_prediction" in proc.stdout
    assert "lesson_46_toy_dialog_reopen_prediction" in proc.stdout
    assert "lesson_47_toy_dialog_resolution_owner_prediction" in proc.stdout
    assert "lesson_48_toy_dialog_resolution_action_prediction" in proc.stdout
    assert "lesson_49_toy_dialog_owner_handoff_prediction" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_01_clip_toy_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_01_clip_toy_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_blip_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_02_blip_toy_captioning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_02_blip_toy_captioning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_llava_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_03_llava_toy_instruction_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_03_llava_toy_instruction_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_grounding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_04_grounding_toy_refexp",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_04_grounding_toy_refexp.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_mask_grounding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_05_mask_grounding_toy_refexp",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_05_mask_grounding_toy_refexp.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_flamingo_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_06_flamingo_toy_interleaved_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_06_flamingo_toy_interleaved_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_qformer_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_07_qformer_toy_bridge_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_07_qformer_toy_bridge_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_perceiver_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_08_perceiver_resampler_toy_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_08_perceiver_resampler_toy_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_paligemma_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_09_paligemma_toy_siglip_decoder_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_09_paligemma_toy_siglip_decoder_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_owlvit_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_10_owlvit_toy_open_vocab_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_10_owlvit_toy_open_vocab_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_grounded_sam_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_11_grounded_sam_toy_open_vocab_segmentation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_11_grounded_sam_toy_open_vocab_segmentation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_key_value_ocr_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_12_key_value_ocr_toy_doc_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_12_key_value_ocr_toy_doc_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_video_vlm_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_13_video_vlm_toy_temporal_qa",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_13_video_vlm_toy_temporal_qa.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_bmn_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_14_bmn_toy_temporal_grounding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_14_bmn_toy_temporal_grounding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_2dtan_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_15_2dtan_toy_temporal_grounding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_15_2dtan_toy_temporal_grounding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_multiscale_2dtan_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_16_multiscale_2dtan_toy_temporal_grounding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_16_multiscale_2dtan_toy_temporal_grounding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_prompt_learning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_18_prompt_learning_vlm",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_18_prompt_learning_vlm.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_text_understanding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_19_audio_text_understanding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_19_audio_text_understanding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_visual_learning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_20_audio_visual_learning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_20_audio_visual_learning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_grounded_retrieval_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_21_audio_grounded_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_21_audio_grounded_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_audio_visual_event_localization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_22_audio_visual_event_localization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_22_audio_visual_event_localization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_embodied_question_answering_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_23_embodied_question_answering",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_23_embodied_question_answering.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_24_multimodal_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_24_multimodal_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_vision_language_navigation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_25_vision_language_navigation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_25_vision_language_navigation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_image_text_reranking_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_26_image_text_reranking",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_26_image_text_reranking.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_scene_text_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_27_scene_text_vlm_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_27_scene_text_vlm_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_document_vlm_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_28_document_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_28_document_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_human_object_interaction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_29_human_object_interaction_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_29_human_object_interaction_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_gaze_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_30_vision_language_gaze_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_30_vision_language_gaze_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_person_search_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_31_person_search_attribute_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_31_person_search_attribute_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_action_localization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_32_video_text_action_localization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_32_video_text_action_localization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_pedestrian_attributes_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_33_pedestrian_attribute_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_33_pedestrian_attribute_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_action_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_34_video_text_action_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_34_video_text_action_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_mamba_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_03_toy_mamba_language_model",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_03_toy_mamba_language_model.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_instruction_tuning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_04_toy_instruction_tuning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_04_toy_instruction_tuning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_prefix_tuning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_05_toy_prefix_tuning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_05_toy_prefix_tuning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_preference_optimization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_06_toy_preference_optimization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_06_toy_preference_optimization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_reward_modeling_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_07_toy_reward_modeling",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_07_toy_reward_modeling.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_span_corruption_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_08_toy_span_corruption",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_08_toy_span_corruption.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_rlhf_ppo_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_09_toy_rlhf_ppo",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_09_toy_rlhf_ppo.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_grpo_alignment_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_10_toy_grpo_alignment",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_10_toy_grpo_alignment.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_rag_language_model_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_11_toy_rag_language_model",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_11_toy_rag_language_model.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_transformer_interpretability_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_12_toy_transformer_interpretability",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_12_toy_transformer_interpretability.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_tool_calling_agent_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_13_toy_tool_calling_agent",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_13_toy_tool_calling_agent.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_replaced_token_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_14_toy_replaced_token_detection_transformer",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_14_toy_replaced_token_detection_transformer.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_judge_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_15_toy_llm_judge",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_15_toy_llm_judge.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_multi_turn_memory_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_16_toy_multi_turn_memory_sft",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_16_toy_multi_turn_memory_sft.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_self_refine_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_17_toy_self_refine_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_17_toy_self_refine_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_reflection_memory_agent_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_18_toy_reflection_memory_agent",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_18_toy_reflection_memory_agent.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_plan_execute_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_19_toy_plan_execute_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_19_toy_plan_execute_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_latent_diffusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_04_toy_latent_diffusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_04_toy_latent_diffusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_consistency_model_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_05_toy_consistency_model",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_05_toy_consistency_model.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_flow_matching_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_06_toy_flow_matching",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_06_toy_flow_matching.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_rectified_flow_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_07_toy_rectified_flow",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_07_toy_rectified_flow.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_transformer_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_08_toy_diffusion_transformer",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_08_toy_diffusion_transformer.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_conditional_gan_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_09_toy_conditional_gan",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_09_toy_conditional_gan.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_image_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_10_toy_diffusion_image_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_10_toy_diffusion_image_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_controlnet_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_11_toy_controlnet",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_11_toy_controlnet.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_to_image_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_12_toy_layout_to_image",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_12_toy_layout_to_image.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_text_to_image_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_13_toy_text_to_image_diffusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_13_toy_text_to_image_diffusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_inpainting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_14_toy_diffusion_inpainting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_14_toy_diffusion_inpainting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_super_resolution_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_15_toy_diffusion_super_resolution",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_15_toy_diffusion_super_resolution.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_deblurring_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_16_toy_diffusion_deblurring",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_16_toy_diffusion_deblurring.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_denoising_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_17_toy_diffusion_denoising",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_17_toy_diffusion_denoising.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_deraining_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_18_toy_diffusion_deraining",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_18_toy_diffusion_deraining.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_dehazing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_19_toy_diffusion_dehazing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_19_toy_diffusion_dehazing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_diffusion_reflection_removal_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_20_toy_diffusion_reflection_removal",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_20_toy_diffusion_reflection_removal.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_transformer_summarization_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_09_toy_transformer_summarization",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_09_toy_transformer_summarization.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_prompt_tuning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_10_toy_prompt_tuning_classifier",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_10_toy_prompt_tuning_classifier.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_few_shot_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_11_toy_few_shot_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_11_toy_few_shot_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_in_context_text_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_12_toy_in_context_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_12_toy_in_context_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_masked_language_modeling_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_13_toy_masked_language_modeling",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_13_toy_masked_language_modeling.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_contrastive_sentence_embedding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_14_toy_contrastive_sentence_embedding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_14_toy_contrastive_sentence_embedding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_cross_encoder_reranking_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_15_toy_cross_encoder_reranking",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_15_toy_cross_encoder_reranking.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_text_clustering_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_16_toy_text_clustering",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_16_toy_text_clustering.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_text_anomaly_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_17_toy_text_anomaly_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_17_toy_text_anomaly_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_topic_modeling_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_18_toy_topic_modeling",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_18_toy_topic_modeling.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_distilled_text_classifier_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_19_toy_distilled_text_classifier",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_19_toy_distilled_text_classifier.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_adversarial_text_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_20_toy_adversarial_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_20_toy_adversarial_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_adversarial_example_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_21_toy_adversarial_example_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_21_toy_adversarial_example_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_weak_supervision_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_22_toy_weak_supervision_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_22_toy_weak_supervision_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_sentence_denoising_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_23_toy_sentence_denoising_autoencoder",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_23_toy_sentence_denoising_autoencoder.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_meta_few_shot_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_24_toy_meta_few_shot_text_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_24_toy_meta_few_shot_text_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_low_shot_intent_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_25_toy_low_shot_intent_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_25_toy_low_shot_intent_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_monocular_depth_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_19_synthetic_monocular_depth_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_19_synthetic_monocular_depth_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_lane_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_20_synthetic_lane_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_20_synthetic_lane_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_lane_topology_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_21_synthetic_lane_topology_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_21_synthetic_lane_topology_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_road_scene_understanding_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_22_synthetic_road_scene_understanding",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_22_synthetic_road_scene_understanding.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_image_dehazing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_23_synthetic_image_dehazing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_23_synthetic_image_dehazing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_reflection_removal_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_24_synthetic_reflection_removal",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_24_synthetic_reflection_removal.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_image_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_25_synthetic_image_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_25_synthetic_image_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_text_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_26_synthetic_text_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_26_synthetic_text_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_edge_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_27_synthetic_edge_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_27_synthetic_edge_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_salient_object_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_28_synthetic_salient_object_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_28_synthetic_salient_object_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_camouflaged_object_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_29_synthetic_camouflaged_object_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_29_synthetic_camouflaged_object_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_salient_object_boxes_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_30_synthetic_salient_object_detection_boxes",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_30_synthetic_salient_object_detection_boxes.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_interactive_segmentation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_31_synthetic_interactive_segmentation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_31_synthetic_interactive_segmentation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_landmark_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_32_synthetic_face_landmark_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_32_synthetic_face_landmark_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_liveness_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_33_synthetic_face_liveness_detection",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_33_synthetic_face_liveness_detection.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_license_plate_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_34_synthetic_license_plate_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_34_synthetic_license_plate_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_pose_6d_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_35_synthetic_6d_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_35_synthetic_6d_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_alignment_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_45_face_alignment_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_45_face_alignment_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_detection_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_46_face_detection_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_46_face_detection_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_retrieval_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_47_face_retrieval_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_47_face_retrieval_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_pose_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_48_face_pose_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_48_face_pose_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_face_gaze_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_49_face_gaze_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_49_face_gaze_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_person_pose_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_50_person_pose_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_50_person_pose_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_hand_pose_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_51_hand_pose_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_51_hand_pose_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_gesture_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_52_gesture_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_52_gesture_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_finger_count_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_53_finger_count_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_53_finger_count_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_handedness_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_54_handedness_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_54_handedness_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_palm_orientation_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_55_palm_orientation_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_55_palm_orientation_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_sign_digit_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_56_sign_digit_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_56_sign_digit_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_finger_spread_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_57_finger_spread_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_57_finger_spread_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_multimodal_thumb_position_reasoning_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "multimodal",
            "lesson_58_thumb_position_vlm_reasoning",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.multimodal.lesson_58_thumb_position_vlm_reasoning.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_citation_grounded_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_30_toy_citation_grounded_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_30_toy_citation_grounded_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_schema_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_31_toy_schema_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_31_toy_schema_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_json_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_32_toy_json_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_32_toy_json_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_function_signature_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_33_toy_function_signature_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_33_toy_function_signature_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_xml_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_34_toy_xml_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_34_toy_xml_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_regex_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_35_toy_regex_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_35_toy_regex_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_ebnf_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_36_toy_ebnf_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_36_toy_ebnf_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_sql_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_37_toy_sql_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_37_toy_sql_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_yaml_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_38_toy_yaml_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_38_toy_yaml_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_csv_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_39_toy_csv_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_39_toy_csv_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_toml_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_40_toy_toml_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_40_toy_toml_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_markdown_table_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_41_toy_markdown_table_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_41_toy_markdown_table_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_ini_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_42_toy_ini_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_42_toy_ini_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_llm_tsv_constrained_prompting_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "llm",
            "lesson_43_toy_tsv_constrained_prompting",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.llm.lesson_43_toy_tsv_constrained_prompting.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_reference_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_31_toy_diffusion_reference_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_31_toy_diffusion_reference_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_preserving_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_32_toy_diffusion_layout_preserving_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_32_toy_diffusion_layout_preserving_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_masked_reference_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_33_toy_diffusion_masked_reference_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_33_toy_diffusion_masked_reference_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_reference_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_34_toy_diffusion_layout_reference_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_34_toy_diffusion_layout_reference_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_box_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_35_toy_diffusion_box_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_35_toy_diffusion_box_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_subject_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_36_toy_diffusion_layout_subject_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_36_toy_diffusion_layout_subject_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_polygon_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_37_toy_diffusion_polygon_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_37_toy_diffusion_polygon_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_attribute_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_38_toy_diffusion_layout_attribute_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_38_toy_diffusion_layout_attribute_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_scribble_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_39_toy_diffusion_scribble_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_39_toy_diffusion_scribble_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_style_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_40_toy_diffusion_layout_style_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_40_toy_diffusion_layout_style_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_stroke_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_41_toy_diffusion_stroke_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_41_toy_diffusion_stroke_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_palette_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_42_toy_diffusion_layout_palette_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_42_toy_diffusion_layout_palette_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_path_mask_editing_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_43_toy_diffusion_path_mask_editing",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_43_toy_diffusion_path_mask_editing.train" in proc.stdout


def test_run_lesson_dry_run_resolves_generative_layout_lighting_fusion_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "generative",
            "lesson_44_toy_diffusion_layout_lighting_fusion",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.generative.lesson_44_toy_diffusion_layout_lighting_fusion.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_retrieval_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_46_synthetic_face_retrieval",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_46_synthetic_face_retrieval.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_face_pose_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_47_synthetic_face_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_47_synthetic_face_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_gaze_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_48_synthetic_gaze_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_48_synthetic_gaze_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_human_pose_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_49_synthetic_human_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_49_synthetic_human_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_hand_pose_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_50_synthetic_hand_pose_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_50_synthetic_hand_pose_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_gesture_recognition_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_51_synthetic_gesture_recognition",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_51_synthetic_gesture_recognition.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_finger_count_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_52_synthetic_finger_count_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_52_synthetic_finger_count_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_handedness_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_53_synthetic_handedness_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_53_synthetic_handedness_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_palm_orientation_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_54_synthetic_palm_orientation_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_54_synthetic_palm_orientation_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_sign_digit_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_55_synthetic_sign_digit_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_55_synthetic_sign_digit_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_finger_spread_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_56_synthetic_finger_spread_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_56_synthetic_finger_spread_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_thumb_position_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_57_synthetic_thumb_position_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_57_synthetic_thumb_position_classification.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_finger_curvature_estimation_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_58_synthetic_finger_curvature_estimation",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_58_synthetic_finger_curvature_estimation.train" in proc.stdout


def test_run_lesson_dry_run_resolves_vision_thumb_contact_classification_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "vision",
            "lesson_59_synthetic_thumb_contact_classification",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.vision.lesson_59_synthetic_thumb_contact_classification.train" in proc.stdout


@pytest.mark.parametrize(
    ("track", "lesson", "train_module"),
    [
        ("vision", "lesson_60_synthetic_image_deraining", "tracks.vision.lesson_60_synthetic_image_deraining.train"),
        ("vision", "lesson_61_synthetic_image_retrieval", "tracks.vision.lesson_61_synthetic_image_retrieval.train"),
        ("vision", "lesson_62_synthetic_image_matching", "tracks.vision.lesson_62_synthetic_image_matching.train"),
        ("vision", "lesson_63_synthetic_image_stitching", "tracks.vision.lesson_63_synthetic_image_stitching.train"),
        (
            "vision",
            "lesson_64_synthetic_fine_grained_recognition",
            "tracks.vision.lesson_64_synthetic_fine_grained_recognition.train",
        ),
        (
            "vision",
            "lesson_65_synthetic_few_shot_recognition",
            "tracks.vision.lesson_65_synthetic_few_shot_recognition.train",
        ),
        (
            "vision",
            "lesson_66_synthetic_video_object_detection",
            "tracks.vision.lesson_66_synthetic_video_object_detection.train",
        ),
        (
            "vision",
            "lesson_67_synthetic_video_stabilization",
            "tracks.vision.lesson_67_synthetic_video_stabilization.train",
        ),
        (
            "vision",
            "lesson_68_synthetic_video_frame_interpolation",
            "tracks.vision.lesson_68_synthetic_video_frame_interpolation.train",
        ),
        (
            "vision",
            "lesson_69_synthetic_video_restoration",
            "tracks.vision.lesson_69_synthetic_video_restoration.train",
        ),
        (
            "vision",
            "lesson_70_synthetic_video_understanding",
            "tracks.vision.lesson_70_synthetic_video_understanding.train",
        ),
        (
            "vision",
            "lesson_71_synthetic_video_summarization",
            "tracks.vision.lesson_71_synthetic_video_summarization.train",
        ),
        (
            "vision",
            "lesson_72_synthetic_video_enhancement",
            "tracks.vision.lesson_72_synthetic_video_enhancement.train",
        ),
        (
            "vision",
            "lesson_73_synthetic_video_object_segmentation",
            "tracks.vision.lesson_73_synthetic_video_object_segmentation.train",
        ),
        (
            "vision",
            "lesson_74_synthetic_video_instance_segmentation",
            "tracks.vision.lesson_74_synthetic_video_instance_segmentation.train",
        ),
        (
            "vision",
            "lesson_75_synthetic_video_matting",
            "tracks.vision.lesson_75_synthetic_video_matting.train",
        ),
        (
            "vision",
            "lesson_76_synthetic_image_deweathering",
            "tracks.vision.lesson_76_synthetic_image_deweathering.train",
        ),
        (
            "vision",
            "lesson_77_synthetic_transparent_depth_estimation",
            "tracks.vision.lesson_77_synthetic_transparent_depth_estimation.train",
        ),
        (
            "vision",
            "lesson_78_synthetic_image_relighting",
            "tracks.vision.lesson_78_synthetic_image_relighting.train",
        ),
        (
            "vision",
            "lesson_79_synthetic_transparent_object_segmentation",
            "tracks.vision.lesson_79_synthetic_transparent_object_segmentation.train",
        ),
        (
            "vision",
            "lesson_80_synthetic_event_camera_understanding",
            "tracks.vision.lesson_80_synthetic_event_camera_understanding.train",
        ),
        (
            "vision",
            "lesson_81_synthetic_shadow_detection",
            "tracks.vision.lesson_81_synthetic_shadow_detection.train",
        ),
        (
            "vision",
            "lesson_82_synthetic_layout_generation",
            "tracks.vision.lesson_82_synthetic_layout_generation.train",
        ),
        (
            "vision",
            "lesson_83_synthetic_panoptic_segmentation",
            "tracks.vision.lesson_83_synthetic_panoptic_segmentation.train",
        ),
        (
            "vision",
            "lesson_84_synthetic_medical_segmentation",
            "tracks.vision.lesson_84_synthetic_medical_segmentation.train",
        ),
        (
            "vision",
            "lesson_85_synthetic_scene_text_spotting",
            "tracks.vision.lesson_85_synthetic_scene_text_spotting.train",
        ),
        (
            "vision",
            "lesson_86_synthetic_co_segmentation",
            "tracks.vision.lesson_86_synthetic_co_segmentation.train",
        ),
        (
            "vision",
            "lesson_87_synthetic_action_recognition",
            "tracks.vision.lesson_87_synthetic_action_recognition.train",
        ),
        (
            "vision",
            "lesson_88_synthetic_reid",
            "tracks.vision.lesson_88_synthetic_reid.train",
        ),
        (
            "vision",
            "lesson_89_synthetic_anomaly_detection",
            "tracks.vision.lesson_89_synthetic_anomaly_detection.train",
        ),
        (
            "pointcloud",
            "lesson_24_toy_pointcloud_completion",
            "tracks.pointcloud.lesson_24_toy_pointcloud_completion.train",
        ),
        (
            "pointcloud",
            "lesson_25_toy_scene_flow_estimation",
            "tracks.pointcloud.lesson_25_toy_scene_flow_estimation.train",
        ),
        (
            "pointcloud",
            "lesson_26_toy_gaussian_splatting",
            "tracks.pointcloud.lesson_26_toy_gaussian_splatting.train",
        ),
        (
            "pointcloud",
            "lesson_27_toy_3d_object_detection",
            "tracks.pointcloud.lesson_27_toy_3d_object_detection.train",
        ),
        (
            "pointcloud",
            "lesson_28_toy_3d_semantic_segmentation",
            "tracks.pointcloud.lesson_28_toy_3d_semantic_segmentation.train",
        ),
        (
            "pointcloud",
            "lesson_29_toy_3d_instance_segmentation",
            "tracks.pointcloud.lesson_29_toy_3d_instance_segmentation.train",
        ),
        (
            "pointcloud",
            "lesson_30_toy_3d_object_tracking",
            "tracks.pointcloud.lesson_30_toy_3d_object_tracking.train",
        ),
        (
            "pointcloud",
            "lesson_31_toy_open_vocabulary_3d",
            "tracks.pointcloud.lesson_31_toy_open_vocabulary_3d.train",
        ),
        (
            "pointcloud",
            "lesson_32_toy_pointcloud_forecasting",
            "tracks.pointcloud.lesson_32_toy_pointcloud_forecasting.train",
        ),
        (
            "pointcloud",
            "lesson_33_toy_pointcloud_anomaly_detection",
            "tracks.pointcloud.lesson_33_toy_pointcloud_anomaly_detection.train",
        ),
        (
            "pointcloud",
            "lesson_34_toy_pointcloud_upsampling",
            "tracks.pointcloud.lesson_34_toy_pointcloud_upsampling.train",
        ),
        (
            "pointcloud",
            "lesson_35_toy_shape_correspondence_3d",
            "tracks.pointcloud.lesson_35_toy_shape_correspondence_3d.train",
        ),
        (
            "pointcloud",
            "lesson_36_toy_pointcloud_registration",
            "tracks.pointcloud.lesson_36_toy_pointcloud_registration.train",
        ),
        (
            "generative",
            "lesson_45_toy_video_diffusion",
            "tracks.generative.lesson_45_toy_video_diffusion.train",
        ),
        (
            "generative",
            "lesson_46_toy_image_to_video_diffusion",
            "tracks.generative.lesson_46_toy_image_to_video_diffusion.train",
        ),
        (
            "generative",
            "lesson_47_toy_text_to_3d",
            "tracks.generative.lesson_47_toy_text_to_3d.train",
        ),
        (
            "generative",
            "lesson_48_toy_image_to_3d",
            "tracks.generative.lesson_48_toy_image_to_3d.train",
        ),
        (
            "generative",
            "lesson_49_toy_text_to_video",
            "tracks.generative.lesson_49_toy_text_to_video.train",
        ),
        (
            "generative",
            "lesson_50_toy_video_to_video",
            "tracks.generative.lesson_50_toy_video_to_video.train",
        ),
        (
            "generative",
            "lesson_51_toy_world_models",
            "tracks.generative.lesson_51_toy_world_models.train",
        ),
    ],
)
def test_run_lesson_dry_run_resolves_batch38_to_batch42_train_modules(
    track: str, lesson: str, train_module: str
) -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/run_lesson.py", track, lesson, "--dry-run"],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert train_module in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_slot_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_36_toy_dialog_slot_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_36_toy_dialog_slot_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_outcome_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_37_toy_dialog_outcome_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_37_toy_dialog_outcome_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_satisfaction_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_38_toy_dialog_satisfaction_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_38_toy_dialog_satisfaction_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_escalation_risk_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_39_toy_dialog_escalation_risk_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_39_toy_dialog_escalation_risk_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_priority_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_40_toy_dialog_priority_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_40_toy_dialog_priority_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_transfer_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_41_toy_dialog_transfer_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_41_toy_dialog_transfer_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_resolution_time_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_42_toy_dialog_resolution_time_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_42_toy_dialog_resolution_time_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_callback_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_43_toy_dialog_callback_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_43_toy_dialog_callback_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_sla_breach_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_44_toy_dialog_sla_breach_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_44_toy_dialog_sla_breach_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_followup_channel_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_45_toy_dialog_followup_channel_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_45_toy_dialog_followup_channel_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_reopen_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_46_toy_dialog_reopen_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_46_toy_dialog_reopen_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_resolution_owner_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_47_toy_dialog_resolution_owner_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_47_toy_dialog_resolution_owner_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_resolution_action_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_48_toy_dialog_resolution_action_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_48_toy_dialog_resolution_action_prediction.train" in proc.stdout


def test_run_lesson_dry_run_resolves_nlp_dialog_owner_handoff_prediction_train_module() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "nlp",
            "lesson_49_toy_dialog_owner_handoff_prediction",
            "--dry-run",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "tracks.nlp.lesson_49_toy_dialog_owner_handoff_prediction.train" in proc.stdout
