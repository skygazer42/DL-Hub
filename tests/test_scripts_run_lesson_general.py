"""Tests for scripts/run_lesson.py: track/lesson listing and generic dry-run behaviour."""

import subprocess
import sys

from _run_lesson_helpers import _repo_root


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


def test_run_lesson_describes_lesson_specific_cli() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_lesson.py",
            "gnn",
            "lesson_01_compact_graph_classification",
            "--describe",
        ],
        cwd=str(_repo_root()),
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "Offline data: built in" in proc.stdout
    assert "\n- --dataset\n" not in proc.stdout
    assert "--epochs" in proc.stdout


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
    assert "lesson_01_clip_compact_retrieval" in proc.stdout
    assert "lesson_02_blip_compact_captioning" in proc.stdout
    assert "lesson_03_llava_compact_instruction_vlm" in proc.stdout
    assert "lesson_04_grounding_compact_refexp" in proc.stdout
    assert "lesson_05_mask_grounding_compact_refexp" in proc.stdout
    assert "lesson_06_flamingo_compact_interleaved_vlm" in proc.stdout
    assert "lesson_07_qformer_compact_bridge_vlm" in proc.stdout
    assert "lesson_08_perceiver_resampler_compact_vlm" in proc.stdout
    assert "lesson_09_paligemma_compact_siglip_decoder_vlm" in proc.stdout
    assert "lesson_10_owlvit_compact_open_vocab_detection" in proc.stdout
    assert "lesson_11_grounded_sam_compact_open_vocab_segmentation" in proc.stdout
    assert "lesson_12_key_value_ocr_compact_doc_vlm" in proc.stdout
    assert "lesson_13_video_vlm_compact_temporal_qa" in proc.stdout
    assert "lesson_14_bmn_compact_temporal_grounding" in proc.stdout
    assert "lesson_15_2dtan_compact_temporal_grounding" in proc.stdout
    assert "lesson_16_multiscale_2dtan_compact_temporal_grounding" in proc.stdout
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
    assert "lesson_01_compact_causal_lm_transformer" in proc.stdout
    assert "lesson_02_compact_chat_sft" in proc.stdout
    assert "lesson_03_compact_mamba_language_model" in proc.stdout
    assert "lesson_04_compact_instruction_tuning" in proc.stdout
    assert "lesson_05_compact_prefix_tuning" in proc.stdout
    assert "lesson_06_compact_preference_optimization" in proc.stdout
    assert "lesson_07_compact_reward_modeling" in proc.stdout
    assert "lesson_08_compact_span_corruption" in proc.stdout
    assert "lesson_09_compact_rlhf_ppo" in proc.stdout
    assert "lesson_10_compact_grpo_alignment" in proc.stdout
    assert "lesson_11_compact_rag_language_model" in proc.stdout
    assert "lesson_12_compact_transformer_interpretability" in proc.stdout
    assert "lesson_13_compact_tool_calling_agent" in proc.stdout
    assert "lesson_14_compact_replaced_token_detection_transformer" in proc.stdout
    assert "lesson_15_compact_llm_judge" in proc.stdout
    assert "lesson_16_compact_multi_turn_memory_sft" in proc.stdout
    assert "lesson_17_compact_self_refine_prompting" in proc.stdout
    assert "lesson_18_compact_reflection_memory_agent" in proc.stdout
    assert "lesson_19_compact_plan_execute_prompting" in proc.stdout
    assert "lesson_20_compact_react_tool_prompting" in proc.stdout
    assert "lesson_21_compact_tree_of_thought_prompting" in proc.stdout
    assert "lesson_22_compact_self_consistency_prompting" in proc.stdout
    assert "lesson_23_compact_critic_rerank_prompting" in proc.stdout
    assert "lesson_24_compact_debate_prompting" in proc.stdout
    assert "lesson_25_compact_verifier_guided_prompting" in proc.stdout
    assert "lesson_26_compact_process_supervision_prompting" in proc.stdout
    assert "lesson_27_compact_self_correction_prompting" in proc.stdout
    assert "lesson_28_compact_reference_grounded_prompting" in proc.stdout
    assert "lesson_29_compact_constraint_repair_prompting" in proc.stdout
    assert "lesson_30_compact_citation_grounded_prompting" in proc.stdout
    assert "lesson_31_compact_schema_constrained_prompting" in proc.stdout
    assert "lesson_32_compact_json_constrained_prompting" in proc.stdout
    assert "lesson_33_compact_function_signature_prompting" in proc.stdout
    assert "lesson_34_compact_xml_constrained_prompting" in proc.stdout
    assert "lesson_35_compact_regex_constrained_prompting" in proc.stdout
    assert "lesson_36_compact_ebnf_constrained_prompting" in proc.stdout
    assert "lesson_37_compact_sql_constrained_prompting" in proc.stdout
    assert "lesson_38_compact_yaml_constrained_prompting" in proc.stdout
    assert "lesson_39_compact_csv_constrained_prompting" in proc.stdout
    assert "lesson_40_compact_toml_constrained_prompting" in proc.stdout
    assert "lesson_41_compact_markdown_table_constrained_prompting" in proc.stdout
    assert "lesson_42_compact_ini_constrained_prompting" in proc.stdout
    assert "lesson_43_compact_tsv_constrained_prompting" in proc.stdout


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
    assert "lesson_03_compact_diffusion_mnist" in proc.stdout
    assert "lesson_04_compact_latent_diffusion" in proc.stdout
    assert "lesson_05_compact_consistency_model" in proc.stdout
    assert "lesson_06_compact_flow_matching" in proc.stdout
    assert "lesson_07_compact_rectified_flow" in proc.stdout
    assert "lesson_08_compact_diffusion_transformer" in proc.stdout
    assert "lesson_09_compact_conditional_gan" in proc.stdout
    assert "lesson_10_compact_diffusion_image_editing" in proc.stdout
    assert "lesson_11_compact_controlnet" in proc.stdout
    assert "lesson_12_compact_layout_to_image" in proc.stdout
    assert "lesson_13_compact_text_to_image_diffusion" in proc.stdout
    assert "lesson_14_compact_diffusion_inpainting" in proc.stdout
    assert "lesson_15_compact_diffusion_super_resolution" in proc.stdout
    assert "lesson_16_compact_diffusion_deblurring" in proc.stdout
    assert "lesson_17_compact_diffusion_denoising" in proc.stdout
    assert "lesson_18_compact_diffusion_deraining" in proc.stdout
    assert "lesson_19_compact_diffusion_dehazing" in proc.stdout
    assert "lesson_20_compact_diffusion_reflection_removal" in proc.stdout
    assert "lesson_21_compact_diffusion_image_fusion" in proc.stdout
    assert "lesson_22_compact_diffusion_style_transfer" in proc.stdout
    assert "lesson_23_compact_diffusion_multi_focus_fusion" in proc.stdout
    assert "lesson_24_compact_diffusion_image_synthesis" in proc.stdout
    assert "lesson_25_compact_diffusion_compositional_generation" in proc.stdout
    assert "lesson_26_compact_diffusion_image_variation" in proc.stdout
    assert "lesson_27_compact_diffusion_reference_guided_generation" in proc.stdout
    assert "lesson_28_compact_diffusion_subject_driven_generation" in proc.stdout
    assert "lesson_29_compact_diffusion_multi_reference_generation" in proc.stdout
    assert "lesson_30_compact_diffusion_identity_preserving_editing" in proc.stdout
    assert "lesson_31_compact_diffusion_reference_editing" in proc.stdout
    assert "lesson_32_compact_diffusion_layout_preserving_editing" in proc.stdout
    assert "lesson_33_compact_diffusion_masked_reference_editing" in proc.stdout
    assert "lesson_34_compact_diffusion_layout_reference_fusion" in proc.stdout
    assert "lesson_35_compact_diffusion_box_mask_editing" in proc.stdout
    assert "lesson_36_compact_diffusion_layout_subject_fusion" in proc.stdout
    assert "lesson_37_compact_diffusion_polygon_mask_editing" in proc.stdout
    assert "lesson_38_compact_diffusion_layout_attribute_fusion" in proc.stdout
    assert "lesson_39_compact_diffusion_scribble_mask_editing" in proc.stdout
    assert "lesson_40_compact_diffusion_layout_style_fusion" in proc.stdout
    assert "lesson_41_compact_diffusion_stroke_mask_editing" in proc.stdout
    assert "lesson_42_compact_diffusion_layout_palette_fusion" in proc.stdout
    assert "lesson_43_compact_diffusion_path_mask_editing" in proc.stdout
    assert "lesson_44_compact_diffusion_layout_lighting_fusion" in proc.stdout
    assert "lesson_45_compact_video_diffusion" in proc.stdout
    assert "lesson_46_compact_image_to_video_diffusion" in proc.stdout
    assert "lesson_47_compact_text_to_3d" in proc.stdout
    assert "lesson_48_compact_image_to_3d" in proc.stdout
    assert "lesson_49_compact_text_to_video" in proc.stdout
    assert "lesson_50_compact_video_to_video" in proc.stdout
    assert "lesson_51_compact_world_models" in proc.stdout


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
    assert "lesson_01_pointnet_compact_classification" in proc.stdout
    assert "lesson_07_pointnet_compact_reconstruction" in proc.stdout
    assert "lesson_23_pointcloud_selfsupervised_ressl" in proc.stdout
    assert "lesson_24_compact_pointcloud_completion" in proc.stdout
    assert "lesson_25_compact_scene_flow_estimation" in proc.stdout
    assert "lesson_26_compact_gaussian_splatting" in proc.stdout
    assert "lesson_27_compact_3d_object_detection" in proc.stdout
    assert "lesson_28_compact_3d_semantic_segmentation" in proc.stdout
    assert "lesson_29_compact_3d_instance_segmentation" in proc.stdout
    assert "lesson_30_compact_3d_object_tracking" in proc.stdout
    assert "lesson_31_compact_open_vocabulary_3d" in proc.stdout
    assert "lesson_32_compact_pointcloud_forecasting" in proc.stdout
    assert "lesson_33_compact_pointcloud_anomaly_detection" in proc.stdout
    assert "lesson_34_compact_pointcloud_upsampling" in proc.stdout
    assert "lesson_35_compact_shape_correspondence_3d" in proc.stdout
    assert "lesson_36_compact_pointcloud_registration" in proc.stdout


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
    assert "lesson_01_compact_text_classification" in proc.stdout
    assert "lesson_07_reading_comprehension" in proc.stdout
    assert "lesson_08_compact_text_matching_biencoder" in proc.stdout
    assert "lesson_09_compact_transformer_summarization" in proc.stdout
    assert "lesson_10_compact_prompt_tuning_classifier" in proc.stdout
    assert "lesson_11_compact_few_shot_text_classification" in proc.stdout
    assert "lesson_12_compact_in_context_text_classification" in proc.stdout
    assert "lesson_13_compact_masked_language_modeling" in proc.stdout
    assert "lesson_14_compact_contrastive_sentence_embedding" in proc.stdout
    assert "lesson_15_compact_cross_encoder_reranking" in proc.stdout
    assert "lesson_16_compact_text_clustering" in proc.stdout
    assert "lesson_17_compact_text_anomaly_detection" in proc.stdout
    assert "lesson_18_compact_topic_modeling" in proc.stdout
    assert "lesson_19_compact_distilled_text_classifier" in proc.stdout
    assert "lesson_20_compact_adversarial_text_classification" in proc.stdout
    assert "lesson_21_compact_adversarial_example_detection" in proc.stdout
    assert "lesson_22_compact_weak_supervision_text_classification" in proc.stdout
    assert "lesson_23_compact_sentence_denoising_autoencoder" in proc.stdout
    assert "lesson_24_compact_meta_few_shot_text_classification" in proc.stdout
    assert "lesson_25_compact_low_shot_intent_detection" in proc.stdout
    assert "lesson_26_compact_joint_intent_slot_parsing" in proc.stdout
    assert "lesson_27_compact_textual_entailment" in proc.stdout
    assert "lesson_28_compact_semantic_textual_similarity" in proc.stdout
    assert "lesson_29_compact_dialog_state_tracking" in proc.stdout
    assert "lesson_30_compact_dialog_response_selection" in proc.stdout
    assert "lesson_31_compact_slot_carryover_prediction" in proc.stdout
    assert "lesson_32_compact_dialog_act_prediction" in proc.stdout
    assert "lesson_33_compact_dialog_intent_prediction" in proc.stdout
    assert "lesson_34_compact_dialog_policy_prediction" in proc.stdout
    assert "lesson_35_compact_dialog_domain_prediction" in proc.stdout
    assert "lesson_36_compact_dialog_slot_prediction" in proc.stdout
    assert "lesson_37_compact_dialog_outcome_prediction" in proc.stdout
    assert "lesson_38_compact_dialog_satisfaction_prediction" in proc.stdout
    assert "lesson_39_compact_dialog_escalation_risk_prediction" in proc.stdout
    assert "lesson_40_compact_dialog_priority_prediction" in proc.stdout
    assert "lesson_41_compact_dialog_transfer_prediction" in proc.stdout
    assert "lesson_42_compact_dialog_resolution_time_prediction" in proc.stdout
    assert "lesson_43_compact_dialog_callback_prediction" in proc.stdout
    assert "lesson_44_compact_dialog_sla_breach_prediction" in proc.stdout
    assert "lesson_45_compact_dialog_followup_channel_prediction" in proc.stdout
    assert "lesson_46_compact_dialog_reopen_prediction" in proc.stdout
    assert "lesson_47_compact_dialog_resolution_owner_prediction" in proc.stdout
    assert "lesson_48_compact_dialog_resolution_action_prediction" in proc.stdout
    assert "lesson_49_compact_dialog_owner_handoff_prediction" in proc.stdout
