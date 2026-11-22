"""Declarative adapters driven by model JSON metadata.

Provides three reusable adapters that let users integrate new camera-controlled,
action-controlled, or prompt-only/API video models without writing Python adapter
code:

- ``DeclarativePoseAdapter`` (``adapter: "declarative_pose"``)
- ``DeclarativeActionAdapter`` (``adapter: "declarative_action"``)
- ``PromptCameraAdapter`` (``adapter: "prompt_camera"``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from wrbench.actions import parse_camera_script
from wrbench.adapters._utils import (
    adapter_taxonomy_metadata,
    ensure_work_dir,
    model_target_trajectory,
    write_json,
)
from wrbench.adapters.base import register_named_adapter
from wrbench.payload import CameraPayload
from wrbench.prompts.camera_text import (
    MODEL_PROMPT_BUILD_METHODS,
    PRESET_TO_GAP,
    build_prompt_to_send,
    camera_clause,
)
from wrbench.registry import canonical_model_key, model_record
from wrbench.trajectory import CameraTrajectory

_VALID_POSE_TYPES = {"c2w", "w2c"}
_VALID_CONVENTIONS = {"opencv", "opengl", "blender"}
_VALID_POSE_FORMATS = {"npy", "npz", "json", "txt_3x4", "txt_4x4"}
_FORMAT_EXTENSIONS = {
    "npy": "npy",
    "npz": "npz",
    "json": "json",
    "txt_3x4": "txt",
    "txt_4x4": "txt",
}

_DEFAULT_ACTION_MAP: dict[str, Any] = {
    "static": "static",
    "yaw_left": "yaw_left",
    "yaw_right": "yaw_right",
    "pan_left": "pan_left",
    "pan_right": "pan_right",
    "pitch_up": "pitch_up",
    "pitch_down": "pitch_down",
    "roll_left": "roll_left",
    "roll_right": "roll_right",
    "dolly_forward": "dolly_forward",
    "dolly_back": "dolly_back",
    "crane_up": "crane_up",
    "crane_down": "crane_down",
}

_ACTION_ALIASES: dict[str, tuple[str, ...]] = {
    "yaw_left": ("yaw_left", "turn_left", "look_left"),
    "yaw_right": ("yaw_right", "turn_right", "look_right"),
    "pan_left": ("pan_left", "left", "move_left", "strafe_left"),
    "pan_right": ("pan_right", "right", "move_right", "strafe_right"),
    "dolly_forward": ("dolly_forward", "forward", "move_forward"),
    "dolly_back": ("dolly_back", "dolly_backward", "back", "backward", "move_backward"),
    "static": ("static", "idle", "none"),
}


def _merge_spec(record: Any, subkey: str) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    caps = record.capabilities if isinstance(record.capabilities, dict) else {}
    cap_sub = caps.get(subkey)
    if isinstance(cap_sub, dict):
        spec.update(cap_sub)
    contract = record.execution_contract if isinstance(record.execution_contract, dict) else {}
    for k, v in contract.items():
        if k != subkey:
            spec[k] = v
    contract_sub = contract.get(subkey)
    if isinstance(contract_sub, dict):
        spec.update(contract_sub)
    return spec


def _convert_c2w_poses(c2w: np.ndarray, *, convention: str, pose_type: str) -> np.ndarray:
    arr = np.asarray(c2w, dtype=np.float32).copy()
    conv = str(convention or "opencv").strip().lower()
    if conv not in _VALID_CONVENTIONS:
        raise ValueError(f"Unsupported pose convention {convention!r}; expected one of {sorted(_VALID_CONVENTIONS)}")
    if conv in {"opengl", "blender"}:
        basis = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        arr[:, :3, :3] = arr[:, :3, :3] @ basis
    ptype = str(pose_type or "c2w").strip().lower()
    if ptype not in _VALID_POSE_TYPES:
        raise ValueError(f"Unsupported pose_type {pose_type!r}; expected one of {sorted(_VALID_POSE_TYPES)}")
    if ptype == "w2c":
        arr = np.linalg.inv(arr).astype(np.float32)
    return arr


def _infer_profile(target: CameraTrajectory) -> str:
    camera_type = str(target.camera_type or "").strip()
    if camera_type in {"static", "yaw_LR", "yaw_RL", "pan_LR", "pan_RL"}:
        return camera_type
    c2w = target.to_c2w()
    rel = np.linalg.inv(c2w[0]) @ c2w
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 0, 0]))
    yaw_peak = float(yaw[int(np.argmax(np.abs(yaw)))]) if len(yaw) else 0.0
    trans = rel[:, :3, 3]
    trans_peak = 0.0
    if trans.size:
        norms = np.linalg.norm(trans, axis=1)
        peak_vec = trans[int(np.argmax(norms))]
        trans_peak = float(peak_vec[int(np.argmax(np.abs(peak_vec)))])
    if abs(yaw_peak) > max(abs(trans_peak), 1e-4):
        return "yaw_LR" if yaw_peak < 0 else "yaw_RL"
    if abs(trans_peak) > 1e-4:
        return "pan_LR" if trans_peak < 0 else "pan_RL"
    return "static"


def _infer_camera_motion_for_prompt(target: CameraTrajectory) -> str:
    camera_type = str(target.camera_type or "").strip()
    if camera_type in PRESET_TO_GAP:
        return PRESET_TO_GAP[camera_type]
    profile = _infer_profile(target)
    return PRESET_TO_GAP.get(profile, "static")


def _allocate_counts(weights: list[int], total: int) -> list[int]:
    if not weights:
        return [total]
    weight_sum = sum(max(1, int(w)) for w in weights)
    raw = [max(1, int(w)) * total / weight_sum for w in weights]
    counts = [int(v) for v in raw]
    remainder = total - sum(counts)
    fracs = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for idx in fracs[: max(0, remainder)]:
        counts[idx] += 1
    return counts


def _extract_canonical_segments(target: CameraTrajectory) -> list[tuple[str, int]]:
    total = int(target.frame_count)
    camera_type = str(target.camera_type or "").strip()
    if camera_type and camera_type not in {"static", "yaw_LR", "yaw_RL", "pan_LR", "pan_RL"}:
        try:
            script = parse_camera_script(camera_type, fps=int(target.fps))
            if script.actions:
                raw_actions: list[str] = []
                raw_weights: list[int] = []
                for action in script.actions:
                    if action.kind == "static":
                        label = "static"
                    elif action.kind == "compound":
                        parts = [
                            f"{kind}_{direction}"
                            for kind, direction, _ in getattr(action, "components", ())
                        ]
                        label = parts[0] if parts else "static"
                    else:
                        label = f"{action.kind}_{action.direction}" if action.direction else action.kind
                    raw_actions.append(label)
                    raw_weights.append(int(action.frames or 1))
                counts = _allocate_counts(raw_weights, total)
                return [(act, cnt) for act, cnt in zip(raw_actions, counts) if cnt > 0]
        except ValueError:
            pass

    profile = _infer_profile(target)
    first = total // 2
    second = total - first
    if profile == "static":
        return [("static", total)]
    if profile == "yaw_LR":
        return [("yaw_left", first), ("yaw_right", second)]
    if profile == "yaw_RL":
        return [("yaw_right", first), ("yaw_left", second)]
    if profile == "pan_LR":
        return [("pan_left", first), ("pan_right", second)]
    if profile == "pan_RL":
        return [("pan_right", first), ("pan_left", second)]
    return [("static", total)]


def _resample_segments_to_schedule(
    base_segments: list[tuple[str, int]],
    segment_frames: list[int],
) -> list[tuple[str, int]]:
    per_frame: list[str] = []
    for action, count in base_segments:
        per_frame.extend([action] * int(count))
    if not per_frame:
        return [("static", int(f)) for f in segment_frames]
    total_target = sum(int(f) for f in segment_frames)
    if total_target <= 0:
        raise ValueError("segment_frames must contain positive integers")
    out: list[tuple[str, int]] = []
    cursor = 0
    for count in segment_frames:
        frames = int(count)
        if frames <= 0:
            raise ValueError("segment_frames entries must be positive integers")
        mid_frac = (cursor + frames / 2.0) / float(total_target)
        src_idx = min(len(per_frame) - 1, max(0, int(mid_frac * len(per_frame))))
        out.append((per_frame[src_idx], frames))
        cursor += frames
    return out


def _lookup_action_value(action: str, action_map: dict[str, Any]) -> Any:
    if action in action_map:
        return action_map[action]
    for alias in _ACTION_ALIASES.get(action, ()):
        if alias in action_map:
            return action_map[alias]
    if "default" in action_map:
        return action_map["default"]
    if "static" in action_map and action == "static":
        return action_map["static"]
    raise ValueError(f"Unmapped canonical action {action!r}; available keys: {sorted(action_map)}")


def _is_numeric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, (list, tuple)) and value:
        return all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in value)
    return False


class DeclarativePoseAdapter:
    """Compile canonical OpenCV C2W trajectories into native pose files from JSON config."""

    name = "declarative_pose"

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
        record = model_record(key)
        spec = _merge_spec(record, "pose_spec")

        pose_type = str(spec.get("pose_type") or "c2w").strip().lower()
        convention = str(spec.get("convention") or "opencv").strip().lower()
        fmt = str(spec.get("format") or "npy").strip().lower()
        if fmt not in _VALID_POSE_FORMATS:
            raise ValueError(f"{key}: unsupported declarative_pose format {fmt!r}; expected {sorted(_VALID_POSE_FORMATS)}")
        npz_key = str(spec.get("npz_key") or "poses").strip() or "poses"
        include_intrinsics = bool(spec.get("include_intrinsics", True))

        target, amp = model_target_trajectory(trajectory, key, num_frames)
        poses = _convert_c2w_poses(target.to_c2w(), convention=convention, pose_type=pose_type)

        pose_file: Path | None = None
        if work_dir is not None:
            out_dir = ensure_work_dir(work_dir)
            filename = str(spec.get("filename") or f"{key.replace('-', '_')}_poses.{_FORMAT_EXTENSIONS[fmt]}")
            pose_file = out_dir / filename
            if fmt == "npy":
                np.save(pose_file, poses)
            elif fmt == "npz":
                npz_payload: dict[str, Any] = {npz_key: poses}
                if include_intrinsics and npz_key != "intrinsics":
                    npz_payload["intrinsics"] = target.intrinsics
                np.savez(pose_file, **npz_payload)
            elif fmt == "json":
                doc: dict[str, Any] = {
                    "pose_type": pose_type,
                    "convention": convention,
                    "poses": poses.astype(float).tolist(),
                }
                if include_intrinsics:
                    doc["intrinsics"] = target.intrinsics.astype(float).tolist()
                write_json(pose_file, doc)
            elif fmt == "txt_3x4":
                rows = poses[:, :3, :4].reshape(poses.shape[0], 12)
                np.savetxt(pose_file, rows, fmt="%.8f")
            elif fmt == "txt_4x4":
                rows = poses.reshape(poses.shape[0], 16)
                np.savetxt(pose_file, rows, fmt="%.8f")

        payload_type = record.payload_type or f"declarative_pose_{fmt}"
        entrypoint = str(spec.get("official_camera_entrypoint") or f"declarative_pose:{pose_type}:{convention}:{fmt}")
        payload_dict: dict[str, Any] = {
            "poses": poses,
            "intrinsics": target.intrinsics,
            "pose_type": pose_type,
            "convention": convention,
            "format": fmt,
            "pose_file": str(pose_file) if pose_file is not None else None,
        }
        if fmt == "npz":
            payload_dict["npz_key"] = npz_key

        return CameraPayload(
            payload_type=payload_type,
            payload=payload_dict,
            target_trajectory=target,
            official_camera_entrypoint=entrypoint,
            coordinate_notes=(
                f"Declarative pose adapter converted canonical OpenCV C2W to "
                f"{convention} {pose_type} ({fmt})."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="direct_frame_pose_payload",
                model_payload_summary={
                    "pose_type": pose_type,
                    "convention": convention,
                    "format": fmt,
                    "pose_count": int(poses.shape[0]),
                    "pose_file": str(pose_file) if pose_file is not None else None,
                },
                control_sample_kind="dense_pose",
                control_sample_count=int(poses.shape[0]),
                sampling_rule=f"declarative_pose_{pose_type}_{convention}_{fmt}",
            ),
        )


class DeclarativeActionAdapter:
    """Compile canonical trajectories into discrete action token timelines or numeric action vectors."""

    name = "declarative_action"

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
        record = model_record(key)
        spec = _merge_spec(record, "action_spec")

        raw_map = spec.get("action_map")
        if raw_map is None:
            action_map = dict(_DEFAULT_ACTION_MAP)
        elif isinstance(raw_map, dict) and raw_map:
            action_map = dict(raw_map)
        else:
            raise ValueError(f"{key}: action_map must be a non-empty JSON object")

        target, amp = model_target_trajectory(trajectory, key, num_frames)
        base_segments = _extract_canonical_segments(target)

        raw_segment_frames = spec.get("segment_frames")
        if raw_segment_frames is not None:
            if not isinstance(raw_segment_frames, list) or not raw_segment_frames:
                raise ValueError(f"{key}: segment_frames must be a non-empty list of positive integers")
            schedule_segments = _resample_segments_to_schedule(base_segments, [int(x) for x in raw_segment_frames])
        else:
            schedule_segments = base_segments

        granularity = str(spec.get("granularity") or ("per_segment" if raw_segment_frames is not None else "per_frame")).strip().lower()
        segments_payload: list[dict[str, Any]] = []
        per_frame_values: list[Any] = []
        per_segment_values: list[Any] = []
        for idx, (canonical_action, count) in enumerate(schedule_segments):
            mapped = _lookup_action_value(canonical_action, action_map)
            segments_payload.append(
                {
                    "segment_index": idx,
                    "action": canonical_action,
                    "value": mapped,
                    "frames": int(count),
                }
            )
            per_segment_values.append(mapped)
            per_frame_values.extend([mapped] * int(count))

        timeline = per_segment_values if granularity == "per_segment" else per_frame_values
        is_numeric = bool(timeline) and all(_is_numeric_value(v) for v in timeline)
        action_array = np.asarray(timeline, dtype=np.float32) if is_numeric else None

        actions_json: Path | None = None
        actions_npy: Path | None = None
        if work_dir is not None:
            out_dir = ensure_work_dir(work_dir)
            actions_json = write_json(
                out_dir / f"{key.replace('-', '_')}_actions.json",
                {
                    "schema_version": 1,
                    "model": key,
                    "profile": _infer_profile(target),
                    "granularity": granularity,
                    "segments": segments_payload,
                    "timeline": timeline,
                },
            )
            if action_array is not None:
                actions_npy = out_dir / f"{key.replace('-', '_')}_actions.npy"
                np.save(actions_npy, action_array)

        payload_type = record.payload_type or "declarative_action"
        payload_dict: dict[str, Any] = {
            "profile": _infer_profile(target),
            "granularity": granularity,
            "segments": segments_payload,
            "timeline": timeline,
            "actions_json": str(actions_json) if actions_json is not None else None,
        }
        if action_array is not None:
            payload_dict["action_array"] = action_array
        if actions_npy is not None:
            payload_dict["actions_npy"] = str(actions_npy)

        return CameraPayload(
            payload_type=payload_type,
            payload=payload_dict,
            target_trajectory=target,
            official_camera_entrypoint=str(spec.get("official_camera_entrypoint") or "declarative_action_timeline"),
            coordinate_notes=(
                "Declarative action adapter mapped canonical camera trajectory to "
                f"{granularity} model-native action controls."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="native_action_condition_payload",
                model_payload_summary={
                    "profile": _infer_profile(target),
                    "granularity": granularity,
                    "segment_count": len(segments_payload),
                    "timeline_length": len(timeline),
                    "numeric": is_numeric,
                    "actions_json": str(actions_json) if actions_json is not None else None,
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="action_matrix_or_pose",
                control_sample_count=len(timeline),
                sampling_rule=f"declarative_action_{granularity}",
            ),
        )


class PromptCameraAdapter:
    """Unified prompt-camera adapter for prompt-only T2V and closed API video models."""

    name = "prompt_camera"

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
        prompt: str = "",
    ) -> CameraPayload:
        key = canonical_model_key(model_name)
        record = model_record(key)
        spec = _merge_spec(record, "prompt_spec")

        target, amp = model_target_trajectory(trajectory, key, num_frames)
        camera_motion = _infer_camera_motion_for_prompt(target)
        raw_camera_type = str(target.camera_type or camera_motion).strip() or camera_motion
        base_prompt = str(prompt or "").strip()

        templates = spec.get("prompt_templates")
        prompt_build_model = str(spec.get("prompt_build_model") or "").strip().lower()
        if not prompt_build_model and key in MODEL_PROMPT_BUILD_METHODS:
            prompt_build_model = key

        if isinstance(templates, dict) and templates:
            template = templates.get(raw_camera_type) or templates.get(camera_motion) or templates.get("default")
            if template is None:
                raise ValueError(
                    f"{key}: prompt_templates has no entry for {raw_camera_type!r} / {camera_motion!r}"
                )
            template_str = str(template)
            if "{prompt}" in template_str:
                prompt_to_send = template_str.format(
                    prompt=base_prompt,
                    camera_motion=camera_motion,
                    camera_type=raw_camera_type,
                ).strip()
            else:
                clause = template_str.format(
                    camera_motion=camera_motion,
                    camera_type=raw_camera_type,
                ).strip()
                prompt_to_send = f"{base_prompt} {clause}".strip() if base_prompt else clause
            build_method = "custom_prompt_templates"
        elif prompt_build_model and prompt_build_model in MODEL_PROMPT_BUILD_METHODS:
            if base_prompt:
                prompt_to_send = build_prompt_to_send(base_prompt, camera_motion, model=prompt_build_model)
            else:
                prompt_to_send = camera_clause(camera_motion)
            build_method = MODEL_PROMPT_BUILD_METHODS[prompt_build_model]
        else:
            clause = camera_clause(camera_motion)
            if base_prompt and clause and clause not in base_prompt:
                prompt_to_send = f"{base_prompt} {clause}".strip()
            else:
                prompt_to_send = base_prompt or clause
            build_method = "camera_clause_append"

        prompt_txt: Path | None = None
        if work_dir is not None:
            out_dir = ensure_work_dir(work_dir)
            prompt_txt = out_dir / f"{key.replace('-', '_')}_prompt.txt"
            prompt_txt.write_text(prompt_to_send + "\n", encoding="utf-8")

        payload_type = "prompt_camera"
        return CameraPayload(
            payload_type=payload_type,
            payload={
                "prompt_to_send": prompt_to_send,
                "source_prompt": base_prompt,
                "camera_motion": camera_motion,
                "camera_type": raw_camera_type,
                "prompt_build_method": build_method,
                "prompt_txt": str(prompt_txt) if prompt_txt is not None else None,
            },
            target_trajectory=target,
            official_camera_entrypoint=str(spec.get("official_camera_entrypoint") or "prompt_camera_text"),
            coordinate_notes=(
                "Prompt-camera adapter compiles canonical camera motion into natural-language "
                "or model-templated prompt instructions; target C2W records intended motion."
            ),
            calibration_status=amp.calibration_status,
            metadata=adapter_taxonomy_metadata(
                model_name=key,
                amp=amp,
                target=target,
                requested_frames=int(num_frames),
                payload_type=payload_type,
                certification_kind="prompt_camera_payload",
                model_payload_summary={
                    "camera_motion": camera_motion,
                    "prompt_build_method": build_method,
                    "prompt_length": len(prompt_to_send),
                    "prompt_txt": str(prompt_txt) if prompt_txt is not None else None,
                },
                target_c2w_is_model_effective=False,
                control_sample_kind="prompt_camera_text",
                control_sample_count=1,
                sampling_rule="natural_language_camera_prompt",
            ),
        )


register_named_adapter("declarative_pose", DeclarativePoseAdapter())
register_named_adapter("declarative_action", DeclarativeActionAdapter())
register_named_adapter("prompt_camera", PromptCameraAdapter())

__all__ = [
    "DeclarativeActionAdapter",
    "DeclarativePoseAdapter",
    "PromptCameraAdapter",
]
