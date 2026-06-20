from __future__ import annotations

from pathlib import Path

import numpy as np

from wrbench.adapters._utils import adapter_taxonomy_metadata, ensure_work_dir, model_target_trajectory, write_json
from wrbench.adapters.base import register
from wrbench.contracts import (
    build_command_template,
    require_execution_contract,
    require_int,
    require_mapping,
    require_str,
)
from wrbench.payload import CameraPayload
from wrbench.registry import canonical_model_key
from wrbench.trajectory import CameraTrajectory


_MINWM_YAW_DEG_PER_TOKEN = 3.0


def _latent_frame_count(video_frames: int, *, latent_stride: int) -> int:
    if int(video_frames) <= 0:
        raise ValueError("video_frames must be positive")
    if int(latent_stride) <= 0:
        raise ValueError("latent_stride must be positive")
    return (int(video_frames) - 1) // int(latent_stride) + 1


def _segment(key: str, count: int) -> str:
    return f"{key}*{max(0, int(count))}"


def _go_return(first_key: str, second_key: str, count: int) -> str:
    if int(count) <= 0:
        return _segment(first_key, 0)
    first = (int(count) + 1) // 2
    second = int(count) - first
    if second <= 0:
        return _segment(first_key, first)
    return f"{_segment(first_key, first)},{_segment(second_key, second)}"


def _target_yaw_peak_deg(target: CameraTrajectory) -> float:
    c2w = target.to_c2w()
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    return float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0


def _yaw_tokens_for_peak(
    first_key: str,
    second_key: str,
    *,
    yaw_peak_deg: float,
    action_count: int,
) -> tuple[str, dict[str, object]]:
    available = max(0, int(action_count))
    first = (available + 1) // 2
    second = available - first
    if second > 0:
        trajectory = f"{_segment(first_key, first)},{_segment(second_key, second)}"
    else:
        trajectory = _segment(first_key, first)

    runtime_step = (
        abs(float(yaw_peak_deg)) / float(first)
        if first > 0
        else _MINWM_YAW_DEG_PER_TOKEN
    )
    effective_peak = float(first) * runtime_step
    sign = -1.0 if first_key == "j" else 1.0
    requires_step_patch = abs(runtime_step - _MINWM_YAW_DEG_PER_TOKEN) > 1e-6
    status = "scaled_rotation_step_go_return" if requires_step_patch else "default_rotation_step_go_return"
    return trajectory, {
        "token_mapping_rule": f"{first_key}_then_{second_key}_minwm_yaw_tokens",
        "token_budget_status": status,
        "minwm_default_yaw_deg_per_token": _MINWM_YAW_DEG_PER_TOKEN,
        "runtime_yaw_deg_per_token": runtime_step,
        "rotation_step_scale": runtime_step / _MINWM_YAW_DEG_PER_TOKEN,
        "requires_runtime_rot_step_patch": requires_step_patch,
        "requested_yaw_peak_deg": abs(float(yaw_peak_deg)),
        "requested_yaw_peak_signed_deg": sign * abs(float(yaw_peak_deg)),
        "available_action_tokens": available,
        "effective_yaw_first_leg_tokens": first,
        "effective_yaw_return_tokens": second,
        "effective_yaw_peak_deg": effective_peak,
        "effective_yaw_peak_signed_deg": sign * effective_peak,
    }


def _trajectory_string(target: CameraTrajectory, *, latent_stride: int) -> tuple[str, str, dict[str, object]]:
    """Approximate WRBench camera families with minWM's native motion tokens."""
    action_count = max(0, _latent_frame_count(target.frame_count, latent_stride=latent_stride) - 1)
    camera_type = str(target.camera_type or "")
    if camera_type == "static":
        return _segment("w", 0), "static_anchor_pose_padded_by_minwm", {
            "token_budget_status": "static",
            "available_action_tokens": action_count,
        }
    if camera_type == "yaw_LR":
        yaw_peak = _target_yaw_peak_deg(target)
        trajectory, details = _yaw_tokens_for_peak("j", "l", yaw_peak_deg=yaw_peak, action_count=action_count)
        return trajectory, "yaw_left_then_right_minwm_tokens", details
    if camera_type == "yaw_RL":
        yaw_peak = _target_yaw_peak_deg(target)
        trajectory, details = _yaw_tokens_for_peak("l", "j", yaw_peak_deg=yaw_peak, action_count=action_count)
        return trajectory, "yaw_right_then_left_minwm_tokens", details
    if camera_type == "pan_LR":
        return _go_return("a", "d", action_count), "translate_left_then_right_minwm_tokens", {
            "token_budget_status": "full_native_token_budget",
            "available_action_tokens": action_count,
        }
    if camera_type == "pan_RL":
        return _go_return("d", "a", action_count), "translate_right_then_left_minwm_tokens", {
            "token_budget_status": "full_native_token_budget",
            "available_action_tokens": action_count,
        }

    c2w = target.to_c2w()
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    yaw_peak = float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0
    trans = rel[:, :3, 3]
    peak = trans[int(np.argmax(np.linalg.norm(trans, axis=1)))] if trans.size else np.zeros(3)
    if abs(yaw_peak) >= max(abs(float(peak[0])), abs(float(peak[2])), 1e-6):
        if yaw_peak < 0:
            trajectory, details = _yaw_tokens_for_peak("j", "l", yaw_peak_deg=yaw_peak, action_count=action_count)
        else:
            trajectory, details = _yaw_tokens_for_peak("l", "j", yaw_peak_deg=yaw_peak, action_count=action_count)
        return trajectory, "inferred_yaw_minwm_tokens", details
    if abs(float(peak[0])) >= abs(float(peak[2])):
        trajectory = _go_return("a", "d", action_count) if float(peak[0]) < 0 else _go_return("d", "a", action_count)
        return trajectory, "inferred_lateral_minwm_tokens", {
            "token_budget_status": "full_native_token_budget",
            "available_action_tokens": action_count,
        }
    trajectory = _go_return("w", "s", action_count) if float(peak[2]) > 0 else _go_return("s", "w", action_count)
    return trajectory, "inferred_forward_minwm_tokens", {
        "token_budget_status": "full_native_token_budget",
        "available_action_tokens": action_count,
    }


@register("minwm-hy-action2v")
class MinWMHyAction2VAdapter:
    name = "minwm_hy_action2v"

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
    ) -> CameraPayload:
        key = canonical_model_key(model_name)
        execution = require_execution_contract(key)
        runtime_parameters = require_mapping(execution, "runtime_parameters")
        benchmark_profile = require_mapping(execution, "wrbench_benchmark_profile")
        official_profile = require_mapping(execution, "official_inference_profile")
        benchmark_frames = require_int(benchmark_profile, "video_length")
        benchmark_fps = require_int(benchmark_profile, "fps")
        chunk_latent_frames = require_int(runtime_parameters, "chunk_latent_frames")
        entrypoint = require_str(execution, "entrypoint")
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        if int(target.frame_count) != benchmark_frames:
            raise ValueError(f"{key} requires contract video_length {benchmark_frames}, got {target.frame_count}")
        if int(target.fps) != benchmark_fps:
            raise ValueError(f"{key} requires contract fps {benchmark_fps}, got {target.fps}")
        trajectory_string, mapping_rule, token_details = _trajectory_string(target, latent_stride=chunk_latent_frames)
        latent_frames = _latent_frame_count(target.frame_count, latent_stride=chunk_latent_frames)
        action_token_count = sum(int(part.split("*", 1)[1]) for part in trajectory_string.split(","))

        trajectory_txt = out_dir / "minwm_hy_action2v_trajectory.txt"
        trajectory_txt.write_text(trajectory_string + "\n", encoding="utf-8")
        example_json = write_json(
            out_dir / "minwm_hy_action2v_example.json",
            [
                {
                    "id": 0,
                    "image": "<first_frame_image>",
                    "caption": "<prompt>",
                    "trajectory": trajectory_string,
                }
            ],
        )
        command_template = build_command_template(
            execution,
            values={
                "example_json": str(example_json),
                "output_dir": "<output_dir>",
                "fps": int(target.fps),
                "height": int(height),
                "width": int(width),
                "video_length": int(target.frame_count),
            },
        )
        request_json = write_json(
            out_dir / "minwm_hy_action2v_run_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "model": require_str(runtime_parameters, "model"),
                "hf_revision": require_str(runtime_parameters, "hf_revision"),
                "transformer_subdir": require_str(runtime_parameters, "transformer_subdir"),
                "base_model": require_str(runtime_parameters, "base_model"),
                "input_contract": {
                    "example_json": str(example_json),
                    "image_field": "absolute first-frame image path required before run",
                    "caption_field": "WRBench dynamic-event prompt string",
                    "trajectory_field": "minWM native motion token string",
                    "trajectory_string": trajectory_string,
                    "token_mapping_rule": mapping_rule,
                    "token_mapping_details": token_details,
                },
                "runtime_parameters": runtime_parameters,
                "official_runtime": require_mapping(execution, "official_runtime"),
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
                "wrbench_policy": require_mapping(execution, "wrbench_policy"),
            },
        )

        payload_type = "minwm_hy_action2v_trajectory_json"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "example_json": str(example_json),
                "trajectory_txt": str(trajectory_txt),
                "request_json": str(request_json),
                "trajectory": trajectory_string,
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="HY15/hy15_inference.py --use_camera with per-sample trajectory in example_json",
            coordinate_notes=(
                "minWM HY Action2V does not consume WRBench C2W directly. WRBench camera families are approximated "
                "as native minWM motion tokens: a/d lateral translation, w/s forward/back, j/l yaw, i/k pitch."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="native_motion_token_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "model": require_str(runtime_parameters, "model"),
                    "hf_revision": require_str(runtime_parameters, "hf_revision"),
                    "transformer_subdir": require_str(runtime_parameters, "transformer_subdir"),
                    "official_video_length": require_int(official_profile, "video_length"),
                    "official_fps": require_int(official_profile, "fps"),
                    "trajectory": trajectory_string,
                    "latent_frame_count": latent_frames,
                    "action_token_count": action_token_count,
                    "token_mapping_rule": mapping_rule,
                    "token_mapping_details": token_details,
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="action_matrix_or_pose",
                control_sample_count=max(1, action_token_count),
                sampling_rule="wrbench_camera_family_to_minwm_native_motion_tokens",
                model_control_extra={
                    "trajectory": trajectory_string,
                    "latent_frame_count": latent_frames,
                    "action_token_count": action_token_count,
                    "token_mapping_details": token_details,
                    "target_c2w_is_desired_wrbench_motion": True,
                    "model_effective_camera_requires_empirical_calibration": True,
                },
            ),
        )


@register("minwm-wan-action2v")
class MinWMWanAction2VAdapter:
    """minWM with the Wan 2.1 Action2V backbone.
    """

    name = "minwm_wan_action2v"

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
    ) -> CameraPayload:
        key = canonical_model_key(model_name)
        execution = require_execution_contract(key)
        runtime_parameters = require_mapping(execution, "runtime_parameters")
        benchmark_profile = require_mapping(execution, "wrbench_benchmark_profile")
        official_profile = require_mapping(execution, "official_inference_profile")
        benchmark_frames = require_int(benchmark_profile, "video_length")
        benchmark_fps = require_int(benchmark_profile, "fps")
        chunk_latent_frames = require_int(runtime_parameters, "chunk_latent_frames")
        entrypoint = require_str(execution, "entrypoint")
        out_dir = ensure_work_dir(work_dir)
        target, amp = model_target_trajectory(trajectory, key, num_frames)
        if int(target.frame_count) != benchmark_frames:
            raise ValueError(f"{key} requires contract video_length {benchmark_frames}, got {target.frame_count}")
        if int(target.fps) != benchmark_fps:
            raise ValueError(f"{key} requires contract fps {benchmark_fps}, got {target.fps}")
        trajectory_string, mapping_rule, token_details = _trajectory_string(target, latent_stride=chunk_latent_frames)
        latent_frames = _latent_frame_count(target.frame_count, latent_stride=chunk_latent_frames)
        action_token_count = sum(int(part.split("*", 1)[1]) for part in trajectory_string.split(","))

        # Wan reads the token string from a line-per-sample
        trajectory_path = out_dir / "minwm_wan_action2v_trajectory.txt"
        trajectory_path.write_text(trajectory_string + "\n", encoding="utf-8")
        data_path = out_dir / "minwm_wan_action2v_prompts.txt"
        data_path.write_text("<prompt>\n", encoding="utf-8")

        command_template = build_command_template(
            execution,
            values={
                "config_path": require_str(runtime_parameters, "config_path"),
                "checkpoint_path": require_str(runtime_parameters, "checkpoint_path"),
                "data_path": str(data_path),
                "trajectory_path": str(trajectory_path),
                "output_dir": "<output_dir>",
                "output_folder": "<output_dir>",
            },
        )
        request_json = write_json(
            out_dir / "minwm_wan_action2v_run_request.json",
            {
                "entrypoint": entrypoint,
                "command_template": command_template,
                "model": require_str(runtime_parameters, "model"),
                "hf_revision": require_str(runtime_parameters, "hf_revision"),
                "transformer_subdir": require_str(runtime_parameters, "transformer_subdir"),
                "base_model": require_str(runtime_parameters, "base_model"),
                "input_contract": {
                    "data_path": str(data_path),
                    "trajectory_path": str(trajectory_path),
                    "caption_field": "WRBench dynamic-event prompt string (one per --data_path line)",
                    "image_field": "not consumed: Wan Action2V DMD camera checkpoint is T2V-only (no i2v)",
                    "trajectory_field": "minWM native motion token string (one per --trajectory_path line)",
                    "trajectory_string": trajectory_string,
                    "token_mapping_rule": mapping_rule,
                    "token_mapping_details": token_details,
                },
                "runtime_parameters": runtime_parameters,
                "official_runtime": require_mapping(execution, "official_runtime"),
                "official_inference_profile": official_profile,
                "wrbench_benchmark_profile": benchmark_profile,
                "wrbench_policy": require_mapping(execution, "wrbench_policy"),
            },
        )

        payload_type = "minwm_wan_action2v_trajectory_json"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "data_path": str(data_path),
                "trajectory_txt": str(trajectory_path),
                "request_json": str(request_json),
                "trajectory": trajectory_string,
                "command_template": command_template,
            },
            target_trajectory=target,
            official_camera_entrypoint="Wan21/wan_inference.py (T2V causal rollout) with per-sample tokens in --trajectory_path",
            coordinate_notes=(
                "minWM Wan Action2V does not consume WRBench C2W directly. WRBench camera families are approximated "
                "as native minWM motion tokens: a/d lateral translation, w/s forward/back, j/l yaw, i/k pitch."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="native_motion_token_payload",
                model_payload_summary={
                    "entrypoint": entrypoint,
                    "model": require_str(runtime_parameters, "model"),
                    "hf_revision": require_str(runtime_parameters, "hf_revision"),
                    "transformer_subdir": require_str(runtime_parameters, "transformer_subdir"),
                    "official_video_length": require_int(official_profile, "video_length"),
                    "official_fps": require_int(official_profile, "fps"),
                    "trajectory": trajectory_string,
                    "latent_frame_count": latent_frames,
                    "action_token_count": action_token_count,
                    "token_mapping_rule": mapping_rule,
                    "token_mapping_details": token_details,
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="action_matrix_or_pose",
                control_sample_count=max(1, action_token_count),
                sampling_rule="wrbench_camera_family_to_minwm_native_motion_tokens",
                model_control_extra={
                    "trajectory": trajectory_string,
                    "latent_frame_count": latent_frames,
                    "action_token_count": action_token_count,
                    "token_mapping_details": token_details,
                    "target_c2w_is_desired_wrbench_motion": True,
                    "model_effective_camera_requires_empirical_calibration": True,
                },
            ),
        )
