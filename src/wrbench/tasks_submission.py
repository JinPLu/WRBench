"""Benchmark task export and video submission manifest validation.

Enables researchers and model authors to benchmark any video-generation model
without integrating its inference runtime into WRBench:

1. ``export_benchmark_tasks(...)`` writes a standardized ``tasks.jsonl`` with
   prompts, first frames, camera scripts, and (optionally) compiled per-model
   target pose/payload artifacts.
2. Users run their own model to produce videos and a JSON/JSONL manifest.
3. ``validate_video_submission(...)`` validates the submission manifest before
   running ``wrbench eval run``.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from wrbench.datasets import (
    load_jsonl,
    natural25_first_frame_path,
    natural25_release_path,
    natural25_variants_path,
)
from wrbench.presets import DEFAULT_FRAMES, PRESETS, build_preset
from wrbench.registry import canonical_model_key, model_record
from wrbench.runner import compile_camera

DEFAULT_EXPORT_CAMERA_PRESETS: tuple[str, ...] = ("static", "yaw_LR", "yaw_RL")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VIDEO_PATH_FIELDS: tuple[str, ...] = ("eval_video_path", "scoring_video_path", "path", "video_path")


def _profile_for_preset(preset: str) -> str:
    if preset == "static":
        return "canonical_static"
    if preset in {"yaw_LR", "yaw_RL"}:
        return "canonical_60deg"
    if preset in {"pan_LR", "pan_RL"}:
        return "canonical_pan"
    return preset


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_presets(
    camera_presets: list[str] | None,
    *,
    supports_static: bool,
) -> list[str]:
    if camera_presets is None:
        presets = list(DEFAULT_EXPORT_CAMERA_PRESETS)
        if not supports_static:
            presets = [p for p in presets if p != "static"]
        return presets
    resolved = [str(p).strip() for p in camera_presets if str(p).strip()]
    if not resolved:
        raise ValueError("camera_presets must contain at least one preset name")
    unknown = [p for p in resolved if p not in PRESETS]
    if unknown:
        raise ValueError(f"Unknown camera preset(s): {unknown}; valid: {sorted(PRESETS)}")
    return resolved


def export_benchmark_tasks(
    *,
    out_dir: str | Path,
    model: str | None = None,
    release_id: str | None = None,
    camera_presets: list[str] | None = None,
    limit: int | None = None,
    compile_targets: bool = True,
) -> dict[str, Any]:
    """Export standardized WRBench benchmark tasks to ``<out_dir>/tasks.jsonl``.

    When ``compile_targets=True`` and ``model`` is specified, also compiles
    per-task target poses and model-native payloads under ``<out_dir>/compiled/``.
    """
    if limit is not None and int(limit) < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    record = None
    model_key: str | None = None
    if model is not None and str(model).strip():
        try:
            model_key = canonical_model_key(model)
            record = model_record(model_key)
        except KeyError:
            if compile_targets:
                raise
            model_key = str(model).strip()

    if record is not None and record.is_deferred and compile_targets:
        raise ValueError(
            f"Model {model_key!r} is deferred and cannot compile targets; pass compile_targets=False"
        )

    supports_static = True
    if record is not None and isinstance(record.capabilities, dict):
        supports_static = bool(record.capabilities.get("supports_static", True))

    presets = _resolve_presets(camera_presets, supports_static=supports_static)

    if release_id is not None and str(release_id).strip():
        variants_path = natural25_release_path("variants.local_ti2v_tv2v.jsonl", release_id=str(release_id).strip())
    else:
        variants_path = natural25_variants_path()

    frames = int(record.default_frames) if record is not None else DEFAULT_FRAMES
    compiled_dir = out_root / "compiled"
    should_compile = bool(compile_targets and record is not None)

    exported_rows: list[dict[str, Any]] = []
    for variant in load_jsonl(variants_path):
        if str(variant.get("oov_gap") or "") != "none":
            continue
        variant_id = str(variant["variant_id"])
        family_id = str(variant["family_id"])
        world_state_prompt = str(variant.get("world_state_prompt") or "").strip()
        expected_state = str(variant.get("expected_state") or "").strip()
        base_prompt = str(variant.get("ti2v_prompt") or "").strip()
        first_frame = str(natural25_first_frame_path(family_id))

        for preset in presets:
            if limit is not None and len(exported_rows) >= int(limit):
                break
            task_id = f"{variant_id}__{preset}"
            video_id = f"{model_key}__{task_id}" if model_key else task_id
            script = build_preset(preset, frames=frames)
            camera_script = script.to_string()
            prompt_to_send = base_prompt

            row: dict[str, Any] = {
                "video_id": video_id,
                "task_id": task_id,
                "family_id": family_id,
                "variant_id": variant_id,
                "reasoning_tier": str(variant.get("reasoning_tier") or ""),
                "event_delta": str(variant.get("event_delta") or ""),
                "camera_type": preset,
                "camera_script": camera_script,
                "profile": _profile_for_preset(preset),
                "world_state_prompt": world_state_prompt,
                "expected_state": expected_state,
                "prompt_to_send": prompt_to_send,
                "first_frame_path": first_frame,
            }
            if model_key:
                row["model"] = model_key

            if should_compile and record is not None and model_key is not None:
                task_work_dir = compiled_dir / task_id
                task_work_dir.mkdir(parents=True, exist_ok=True)
                out_stub = compiled_dir / f"{task_id}.mp4"
                compile_kwargs: dict[str, Any] = {
                    "model": model_key,
                    "camera": script,
                    "camera_type": preset,
                    "out": out_stub,
                    "prompt": base_prompt,
                    "work_dir": task_work_dir,
                    "dry_run": True,
                }
                if record.input_kind == "image":
                    compile_kwargs["image"] = first_frame
                elif record.input_kind == "source_video":
                    compile_kwargs["source_video"] = str(task_work_dir / "source_video.mp4")

                compiled = compile_camera(**compile_kwargs)
                artifacts = compiled["artifacts"]
                row["target_pose_path"] = artifacts["target_pose_path"]
                row["payload_json_path"] = artifacts["payload_json"]
                row["camera_sidecar_path"] = artifacts["camera_sidecar_path"]
                payload_data = compiled["payload"].payload
                if isinstance(payload_data, dict) and payload_data.get("prompt_to_send"):
                    row["prompt_to_send"] = str(payload_data["prompt_to_send"])

            exported_rows.append(row)
        if limit is not None and len(exported_rows) >= int(limit):
            break

    tasks_jsonl = out_root / "tasks.jsonl"
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in exported_rows)
    tasks_jsonl.write_text(lines, encoding="utf-8")

    return {
        "ok": True,
        "tasks_count": len(exported_rows),
        "tasks_jsonl": str(tasks_jsonl),
        "out_dir": str(out_root),
    }


def _load_manifest_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    errors: list[str] = []
    if not path.is_file():
        return [], [f"Manifest file not found: {path}"]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return [], [f"Manifest file is empty: {path}"]

    if text[0] in "[{":
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                if not payload:
                    return [], [f"Manifest contains 0 rows: {path}"]
                rows: list[dict[str, Any]] = []
                for idx, item in enumerate(payload):
                    if not isinstance(item, dict):
                        errors.append(f"row {idx}: expected JSON object, got {type(item).__name__}")
                    else:
                        rows.append(item)
                return rows, errors
            if isinstance(payload, dict):
                for wrapper_key in ("records", "rows", "videos", "tasks", "data"):
                    seq = payload.get(wrapper_key)
                    if isinstance(seq, list):
                        if not seq:
                            return [], [f"Manifest '{wrapper_key}' list is empty: {path}"]
                        rows = []
                        for idx, item in enumerate(seq):
                            if not isinstance(item, dict):
                                errors.append(f"row {idx}: expected JSON object, got {type(item).__name__}")
                            else:
                                rows.append(item)
                        return rows, errors
                return [payload], errors
            return [], [f"Manifest JSON root must be a list or object: {path}"]
        except json.JSONDecodeError:
            pass

    rows = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON ({exc})")
            continue
        if not isinstance(item, dict):
            errors.append(f"line {line_no}: expected JSON object, got {type(item).__name__}")
            continue
        rows.append(item)
    if not rows and not errors:
        errors.append(f"Manifest contains 0 rows: {path}")
    return rows, errors


def _resolve_submission_path(raw_path: str, *, video_root: Path | None) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute() and video_root is not None:
        candidate = video_root / candidate
    return candidate


def validate_video_submission(
    manifest_path: str | Path,
    *,
    video_root: str | Path | None = None,
    check_files: bool = True,
    check_sha256: bool = False,
) -> dict[str, Any]:
    """Validate a JSON/JSONL video submission manifest prior to evaluation."""
    manifest = Path(manifest_path).expanduser()
    root = Path(video_root).expanduser() if video_root is not None else None

    rows, errors = _load_manifest_rows(manifest)
    warnings: list[str] = []
    models: set[str] = set()
    seen_video_ids: set[str] = set()

    for idx, row in enumerate(rows):
        raw_vid = row.get("video_id")
        video_id = str(raw_vid).strip() if isinstance(raw_vid, str) else ""
        label = f"row {idx} ({video_id})" if video_id else f"row {idx}"
        if not video_id:
            errors.append(f"row {idx}: missing non-empty 'video_id'")
        elif video_id in seen_video_ids:
            errors.append(f"{label}: duplicate video_id {video_id!r}")
        else:
            seen_video_ids.add(video_id)

        raw_model = row.get("model")
        model_name = str(raw_model).strip() if isinstance(raw_model, str) else ""
        if not model_name:
            errors.append(f"{label}: missing non-empty 'model'")
        else:
            models.add(model_name)

        raw_video_path = ""
        for field in _VIDEO_PATH_FIELDS:
            val = row.get(field)
            if isinstance(val, str) and val.strip():
                raw_video_path = val.strip()
                break

        if not raw_video_path:
            errors.append(f"{label}: missing video path field ('path' or 'video_path')")
        elif check_files or check_sha256:
            resolved_video = _resolve_submission_path(raw_video_path, video_root=root)
            if not resolved_video.is_file():
                errors.append(f"{label}: video file not found: {resolved_video}")
            else:
                if resolved_video.stat().st_size <= 0:
                    errors.append(f"{label}: video file is empty (0 bytes): {resolved_video}")
                if check_sha256:
                    expected_sha = str(row.get("sha256") or row.get("video_sha256") or "").strip().lower()
                    if not _SHA256_RE.fullmatch(expected_sha):
                        errors.append(f"{label}: check_sha256=True requires a valid 64-hex 'sha256' field")
                    else:
                        actual_sha = _sha256_file(resolved_video)
                        if actual_sha != expected_sha:
                            errors.append(
                                f"{label}: sha256 mismatch (expected {expected_sha}, got {actual_sha})"
                            )

        camera_type = str(row.get("camera_type") or "").strip()
        if not camera_type:
            warnings.append(f"{label}: missing 'camera_type' (needed for D1 camera alignment scoring)")

        world_state_prompt = str(row.get("world_state_prompt") or "").strip()
        if not world_state_prompt:
            warnings.append(
                f"{label}: missing 'world_state_prompt' (needed for D3-D6 world-reasoning scoring)"
            )

        target_pose_raw = row.get("target_pose_path")
        if check_files and isinstance(target_pose_raw, str) and target_pose_raw.strip():
            resolved_pose = _resolve_submission_path(target_pose_raw.strip(), video_root=root)
            if not resolved_pose.is_file():
                warnings.append(f"{label}: target_pose_path not found: {resolved_pose}")

    return {
        "ok": len(errors) == 0,
        "row_count": len(rows),
        "errors": errors,
        "warnings": warnings,
        "models": sorted(models),
    }


__all__ = [
    "DEFAULT_EXPORT_CAMERA_PRESETS",
    "export_benchmark_tasks",
    "validate_video_submission",
]
