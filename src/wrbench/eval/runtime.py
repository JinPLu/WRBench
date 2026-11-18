"""Eval runtime configuration and orchestration helpers."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wrbench.eval._io import read_records, write_json, write_jsonl
from wrbench.eval.plugins import DEFAULT_BACKENDS, DEFAULT_D1_SCORER_CONFIG, get_backend
from wrbench.runtime import (
    RuntimeConfigError,
    _load_runtime_payload,
    _optional_mapping,
    _require_int,
    _resolve_runtime_path,
)

D3D6_SCORES = Path("final_exports/scores_v7_runtime_v2_evidence_first_gate_masked_export.json")


@dataclass(frozen=True)
class EvalScorerRuntime:
    """Paths to external model scorers (host-only, not pip deps)."""

    gpu_id: int
    vggt_python_bin: str | None = None
    vggt_repo: str | None = None
    vggt_checkpoint: str | None = None
    dinov2_python_bin: str | None = None
    dinov2_model_path: str | None = None
    qwen_scorer_python: str | None = None
    qwen35_model: str | None = None
    qwen3vl_model: str | None = None
    extra_paths: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRuntimeConfig:
    schema_version: int
    scorers: EvalScorerRuntime
    backends: dict[str, str] = field(default_factory=dict)


def _parse_eval_node(payload: dict[str, Any], *, context: str) -> EvalScorerRuntime:
    eval_block = payload.get("eval")
    if isinstance(eval_block, dict) and isinstance(eval_block.get("scorers"), dict):
        node = eval_block["scorers"]
    else:
        raise RuntimeConfigError(f"{context}: required object field 'eval.scorers' is missing or invalid")
    return EvalScorerRuntime(
        vggt_python_bin=node.get("vggt_python_bin"),
        vggt_repo=node.get("vggt_repo"),
        vggt_checkpoint=node.get("vggt_checkpoint"),
        dinov2_python_bin=node.get("dinov2_python_bin"),
        dinov2_model_path=node.get("dinov2_model_path"),
        qwen_scorer_python=node.get("qwen_scorer_python"),
        qwen35_model=node.get("qwen35_model"),
        qwen3vl_model=node.get("qwen3vl_model"),
        gpu_id=_require_int(node, "gpu_id", context=f"{context}: eval.scorers"),
        extra_paths={str(k): str(v) for k, v in _optional_mapping(node, "extra_paths", context=f"{context}: eval.scorers").items()},
        env={str(k): str(v) for k, v in _optional_mapping(node, "env", context=f"{context}: eval.scorers").items()},
    )


def load_eval_runtime(path: Path | None = None) -> EvalRuntimeConfig | None:
    resolved = _resolve_runtime_path(path)
    if resolved is None:
        return None
    payload = _load_runtime_payload(resolved)
    if "eval" not in payload and "scorers" not in payload:
        return None
    scorers = _parse_eval_node(payload, context=str(resolved))
    backends = _optional_mapping(payload["eval"], "backends", context=f"{resolved}: eval")
    for stage, name in backends.items():
        if stage not in DEFAULT_BACKENDS or not isinstance(name, str) or not name.strip():
            raise RuntimeConfigError(f"{resolved}: eval.backends.{stage} must select a pose, d2, or vlm backend")
    return EvalRuntimeConfig(
        schema_version=int(payload["schema_version"]),
        scorers=scorers,
        backends={stage: name.strip() for stage, name in backends.items()},
    )


def _require_extra(scorers: EvalScorerRuntime, field: str) -> str:
    value = scorers.extra_paths.get(field)
    if value is None or not str(value).strip():
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} is required")
    return str(value).strip()


def _require_extra_int(scorers: EvalScorerRuntime, field: str) -> int:
    value = _require_extra(scorers, field)
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} must be an integer") from exc
    return parsed


def _require_extra_float(scorers: EvalScorerRuntime, field: str) -> float:
    value = _require_extra(scorers, field)
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} must be a number") from exc


def _extra_or_default(scorers: EvalScorerRuntime, field: str, default: str) -> str:
    value = scorers.extra_paths.get(field)
    return str(value).strip() if value is not None and str(value).strip() else str(default)


def _extra_int_or_default(scorers: EvalScorerRuntime, field: str, default: int) -> int:
    raw = scorers.extra_paths.get(field)
    if raw is None or not str(raw).strip():
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} must be an integer") from exc


def _extra_float_or_default(scorers: EvalScorerRuntime, field: str, default: float) -> float:
    raw = scorers.extra_paths.get(field)
    if raw is None or not str(raw).strip():
        return float(default)
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeConfigError(f"eval.scorers.extra_paths.{field} must be a number") from exc


def require_eval_runtime(path: Path | None = None) -> EvalRuntimeConfig:
    cfg = load_eval_runtime(path)
    if cfg is None:
        raise RuntimeError(
            "eval scorers not configured: add an 'eval.scorers' section to wrbench.runtime.json "
            "(see wrbench.runtime.example.json)"
        )
    return cfg


def contract_path() -> Path:
    return Path(__file__).resolve().parent / "contract" / "latest_d1_d6_metric_contract.json"


def run_scorer_python(
    python_bin: str,
    module: str,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    package_root = str(Path(__file__).resolve().parents[2])
    python_path = merged.get("PYTHONPATH")
    merged["PYTHONPATH"] = os.pathsep.join(p for p in (package_root, python_path) if p)
    cmd = [python_bin, "-m", module, *argv]
    proc = subprocess.run(cmd, env=merged)
    return int(proc.returncode)


def run_shell_script(script: Path, stage: str, *, env: dict[str, str]) -> int:
    merged = os.environ.copy()
    merged.update(env)
    proc = subprocess.run(["bash", str(script), stage], env=merged)
    return int(proc.returncode)


def d1_score(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    summary_csv: Path,
    pose_cache_root: Path,
    pose_backend: str = str(DEFAULT_D1_SCORER_CONFIG["pose_backend"]),
    poses_file: str = str(DEFAULT_D1_SCORER_CONFIG["poses_file"]),
    default_frames: int = int(DEFAULT_D1_SCORER_CONFIG["default_frames"]),
    sidecar_profile_gate: str = "main",
    predicted_pose_type: str = str(DEFAULT_D1_SCORER_CONFIG["predicted_pose_type"]),
    predicted_camera_convention: str = str(DEFAULT_D1_SCORER_CONFIG["predicted_camera_convention"]),
    target_camera_convention: str = str(DEFAULT_D1_SCORER_CONFIG["target_camera_convention"]),
    rot_scale_deg: float = float(DEFAULT_D1_SCORER_CONFIG["rot_scale_deg"]),
    trans_scale: float = float(DEFAULT_D1_SCORER_CONFIG["trans_scale"]),
    yaw_weak_threshold_deg: float = float(DEFAULT_D1_SCORER_CONFIG["yaw_weak_threshold_deg"]),
    pan_weak_threshold: float = float(DEFAULT_D1_SCORER_CONFIG["pan_weak_threshold"]),
    static_rot_threshold_deg: float = float(DEFAULT_D1_SCORER_CONFIG["static_rot_threshold_deg"]),
    static_trans_threshold: float = float(DEFAULT_D1_SCORER_CONFIG["static_trans_threshold"]),
    target_fallback_mode: str = str(DEFAULT_D1_SCORER_CONFIG["target_fallback_mode"]),
) -> int:
    from wrbench.eval.d1.d1_camera import main as d1_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-jsonl",
        str(output_jsonl),
        "--summary-csv",
        str(summary_csv),
        "--megasam-cache-root",
        str(pose_cache_root),
        "--pose-backend",
        pose_backend,
        "--poses-file",
        poses_file,
        "--default-frames",
        str(default_frames),
        "--sidecar-profile-gate",
        sidecar_profile_gate,
        "--target-mode",
        target_fallback_mode,
        "--predicted-pose-type",
        predicted_pose_type,
        "--predicted-camera-convention",
        predicted_camera_convention,
        "--target-camera-convention",
        target_camera_convention,
        "--rot-scale-deg",
        str(rot_scale_deg),
        "--trans-scale",
        str(trans_scale),
        "--yaw-weak-threshold-deg",
        str(yaw_weak_threshold_deg),
        "--pan-weak-threshold",
        str(pan_weak_threshold),
        "--static-rot-threshold-deg",
        str(static_rot_threshold_deg),
        "--static-trans-threshold",
        str(static_trans_threshold),
    ]
    return d1_main(argv)


def d1_camalign_score(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    pose_cache_root: Path,
    poses_file: str,
    scorer_version: str = "v2_lite",
    summary_json: Path | None = None,
) -> int:
    from wrbench.eval.d1.d1_camalign import main as camalign_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-jsonl",
        str(output_jsonl),
        "--pose-cache-root",
        str(pose_cache_root),
        "--poses-file",
        poses_file,
        "--scorer-version",
        scorer_version,
    ]
    if summary_json is not None:
        argv.extend(["--summary-json", str(summary_json)])
    return camalign_main(argv)


def d1_vggt_batch(
    *,
    eval_runtime: EvalRuntimeConfig,
    input_jsonl: Path,
    output_root: Path,
    cache_root: Path,
    execution_mode: str | None = None,
    video_root: Path | None = None,
) -> int:
    backend = get_backend("pose", eval_runtime.backends.get("pose"))
    return backend(
        eval_runtime=eval_runtime,
        input_jsonl=input_jsonl,
        output_root=output_root,
        cache_root=cache_root,
        execution_mode=execution_mode,
        video_root=video_root,
    )


def _run_vggt_omega(
    *,
    eval_runtime: EvalRuntimeConfig,
    input_jsonl: Path,
    output_root: Path,
    cache_root: Path,
    execution_mode: str | None,
    video_root: Path | None,
) -> int:
    scorers = eval_runtime.scorers
    for field_name, value in (
        ("vggt_python_bin", scorers.vggt_python_bin),
        ("vggt_repo", scorers.vggt_repo),
        ("vggt_checkpoint", scorers.vggt_checkpoint),
    ):
        if not value or not Path(str(value)).exists():
            raise FileNotFoundError(f"eval.scorers.{field_name} missing or not found: {value!r}")
    from wrbench.eval.d1.run_d1_vggt_omega_batch import main as batch_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-root",
        str(output_root),
        "--vggt-repo",
        str(scorers.vggt_repo),
        "--checkpoint",
        str(scorers.vggt_checkpoint),
        "--gpu-id",
        str(scorers.gpu_id),
        "--cache-root",
        str(cache_root),
        "--python",
        str(scorers.vggt_python_bin),
        "--execution-mode",
        execution_mode or _require_extra(scorers, "d1_vggt_execution_mode"),
        "--poses-file",
        _extra_or_default(scorers, "d1_poses_file", DEFAULT_D1_SCORER_CONFIG["poses_file"]),
        "--image-resolution",
        str(_require_extra_int(scorers, "d1_vggt_image_resolution")),
        "--preprocess-mode",
        _require_extra(scorers, "d1_vggt_preprocess_mode"),
        "--cwd",
        _require_extra(scorers, "d1_vggt_cwd"),
        "--shard-index",
        _require_extra(scorers, "d1_vggt_shard_index"),
        "--num-shards",
        _require_extra(scorers, "d1_vggt_num_shards"),
    ]
    if video_root is not None:
        argv.extend(["--video-root", str(video_root)])
    return batch_main(argv)


def d2_extract(
    *,
    eval_runtime: EvalRuntimeConfig,
    videos_manifest: Path,
    out_jsonl: Path,
    model_dir: Path | None = None,
    video_root: Path | None = None,
) -> int:
    backend = get_backend("d2", eval_runtime.backends.get("d2"))
    return backend(
        eval_runtime=eval_runtime,
        videos_manifest=videos_manifest,
        out_jsonl=out_jsonl,
        model_dir=model_dir,
        video_root=video_root,
    )


def _d2_sampling(scorers: EvalScorerRuntime) -> dict[str, str | float | int]:
    return {
        "sample_policy": _require_extra(scorers, "d2_sample_policy"),
        "sample_fps": _require_extra_float(scorers, "d2_sample_fps"),
        "min_frames": _require_extra_int(scorers, "d2_min_frames"),
        "max_frames": _require_extra_int(scorers, "d2_max_frames"),
        "batch_size": _require_extra_int(scorers, "d2_batch_size"),
    }


def _run_dinov2_fullfov(
    *,
    eval_runtime: EvalRuntimeConfig,
    videos_manifest: Path,
    out_jsonl: Path,
    model_dir: Path | None,
    video_root: Path | None,
) -> int:
    scorers = eval_runtime.scorers
    if not scorers.dinov2_python_bin or not Path(str(scorers.dinov2_python_bin)).exists():
        raise FileNotFoundError(f"eval.scorers.dinov2_python_bin missing or not found: {scorers.dinov2_python_bin!r}")
    if model_dir is None and (not scorers.dinov2_model_path or not Path(str(scorers.dinov2_model_path)).exists()):
        raise FileNotFoundError(f"eval.scorers.dinov2_model_path missing or not found: {scorers.dinov2_model_path!r}")
    python_bin = scorers.dinov2_python_bin
    model = model_dir if model_dir is not None else Path(str(scorers.dinov2_model_path))
    argv = [
        "--videos",
        str(videos_manifest),
        "--out-jsonl",
        str(out_jsonl),
        "--device",
        _require_extra(scorers, "d2_device"),
    ]
    for key, value in _d2_sampling(scorers).items():
        argv.extend(["--" + key.replace("_", "-"), str(value)])
    if video_root is not None:
        argv.extend(["--video-root", str(video_root)])
    argv.extend(["--model-dir", str(model)])
    return run_scorer_python(
        python_bin,
        "wrbench.eval.d2.extract_d2_dinov2_fullfov_candidate",
        argv,
        env={"CUDA_VISIBLE_DEVICES": str(scorers.gpu_id), **scorers.env},
    )


def d3d6_env(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scorer_profile: str,
) -> dict[str, str]:
    profile = str(scorer_profile).strip()
    if not profile:
        raise ValueError("scorer_profile is required")
    scorers = eval_runtime.scorers
    for field_name, value in (
        ("qwen_scorer_python", scorers.qwen_scorer_python),
        ("qwen35_model", scorers.qwen35_model),
        ("qwen3vl_model", scorers.qwen3vl_model),
    ):
        if not value or not Path(str(value)).exists():
            raise FileNotFoundError(f"eval.scorers.{field_name} missing or not found: {value!r}")
    required_env = (
        "FORCE_QWENVL_VIDEO_READER",
        "WORLD_STATE_VIDEO_BACKEND",
        "RUN_TAG",
        "PROMPT_MODE",
        "TASK_CONTEXT_MODE",
        "NUM_SHARDS",
        "SHARD_IDS",
        "CUDA_DEVICES_CSV",
        "FPS",
        "PROGRESS_EVERY",
        "SKIP_EXISTING",
        "QWEN35_VLM_NAME",
        "QWEN35_LOADER_FAMILY",
        "QWEN35_DTYPE",
        "QWEN35_ATTN_IMPLEMENTATION",
        "QWEN35_NUM_SAMPLES",
        "QWEN35_MAX_VIDEOS",
        "QWEN35_EVIDENCE_CONTEXT_MODE",
        "QWEN3VL_DTYPE",
        "QWEN3VL_ATTN_IMPLEMENTATION",
        "QWEN3VL_MAX_NEW_TOKENS",
        "QWEN3VL_MAX_VIDEOS",
        "QWEN3VL_BINARY_PROMPT_SCHEMA",
        "QWEN3VL_RESCUE_PROMPT_SCHEMA",
    )
    missing_env = [name for name in required_env if not str(scorers.env.get(name, "")).strip()]
    if missing_env:
        raise RuntimeConfigError(
            "eval.scorers.env missing required D3-D6 field(s): " + ", ".join(missing_env)
        )
    shell = Path(__file__).with_name("score_runtime_v2_d3d6.sh")
    env = {
        **scorers.env,
        "MANIFEST": str(manifest.resolve()),
        "OUT_DIR": str(out_dir.resolve()),
        "SCORER_PROFILE": profile,
        "PY_SCORER": str(scorers.qwen_scorer_python),
        "PY_HELPER": str(scorers.qwen_scorer_python),
        "QWEN35_MODEL": str(scorers.qwen35_model),
        "QWEN3VL_MODEL": str(scorers.qwen3vl_model),
        "CUDA_VISIBLE_DEVICES": str(scorers.gpu_id),
    }
    if not shell.is_file():
        raise FileNotFoundError(f"D3-D6 shell orchestrator missing: {shell}")
    env["WRBENCH_D3D6_SCRIPT"] = str(shell)
    return env


def d3d6_score(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    stage: str,
    scorer_profile: str,
) -> int:
    selected = eval_runtime.backends.get("vlm")
    if selected and selected != DEFAULT_BACKENDS["vlm"] and scorer_profile == "wrbench_default":
        raise ValueError("Use --scorer-profile custom for a nonstandard VLM backend")
    backend = get_backend("vlm", selected)
    scores_path = out_dir / D3D6_SCORES
    rc = backend(
        eval_runtime=eval_runtime,
        manifest=manifest,
        out_dir=out_dir,
        scores_path=scores_path,
        stage=stage,
        scorer_profile=scorer_profile,
    )
    if rc == 0 and stage in {"all", "export"} and not scores_path.is_file():
        raise FileNotFoundError(f"D3-D6 export missing: {scores_path}")
    return rc


def _run_qwen_runtime(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scores_path: Path,
    stage: str,
    scorer_profile: str,
) -> int:
    env = d3d6_env(
        eval_runtime=eval_runtime,
        manifest=manifest,
        out_dir=out_dir,
        scorer_profile=scorer_profile,
    )
    script = Path(env["WRBENCH_D3D6_SCRIPT"])
    return run_shell_script(script, stage, env=env)


def _resolve_manifest_paths(row: dict[str, Any], base: Path) -> dict[str, Any]:
    resolved = dict(row)
    for key in (
        "path", "video_path", "eval_video_path", "scoring_video_path", "original_concat_path",
        "source_video_path", "target_pose_path", "input_pose_path", "trajectory_c2w_path",
    ):
        if row.get(key):
            resolved[key] = str(base / str(row[key]))
    if resolved.get("evaluated_segment") == "generated_only" and resolved.get("original_concat_path"):
        resolved.setdefault("scoring_video_surface", "full_continuation")
    return resolved


def eval_run(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scorer_profile: str,
    sidecar_profile_gate: str,
    target_fallback_mode: str = "none",
    video_root: Path | None = None,
    camalign_scorer_version: str = "v2_lite",
) -> int:
    """Run the full WRBench eval pipeline: D1 pose -> D1 score -> D2 -> D3-D6 -> table."""
    if scorer_profile == "wrbench_default":
        from wrbench.eval.d2.extract_d2_dinov2_fullfov_candidate import PAPER_SAMPLING

        if any(name != DEFAULT_BACKENDS[stage] for stage, name in eval_runtime.backends.items()):
            raise ValueError("Use --scorer-profile custom when replacing a paper evaluation backend")
        if camalign_scorer_version != "v2_lite":
            raise ValueError("wrbench_default requires CamAlign v2_lite. Use --scorer-profile custom for v1.")
        if _d2_sampling(eval_runtime.scorers) != PAPER_SAMPLING:
            raise ValueError(
                "wrbench_default requires D2 time_fps, 3 fps, min 4, max 24, batch 4. "
                "Update eval.scorers.extra_paths from the runtime example or use --scorer-profile custom."
            )
    for stage, name in eval_runtime.backends.items():
        get_backend(stage, name)

    out_dir = out_dir.resolve()
    manifest = manifest.resolve()
    base = video_root.resolve() if video_root is not None else Path.cwd()
    rows = [_resolve_manifest_paths(row, base) for row in read_records(manifest, strict=True)]
    video_ids = [str(row.get("video_id") or "").strip() for row in rows]
    if not video_ids or "" in video_ids or len(set(video_ids)) != len(video_ids):
        raise ValueError("The manifest must contain at least one video and unique, nonempty video_id values")
    manifest_json = out_dir / "manifest.resolved.json"
    if manifest_json == manifest:
        manifest_json = out_dir / "manifest.resolved.input.json"
    write_json(manifest_json, rows)

    d1_input = out_dir / "d1_input.jsonl"
    write_jsonl(rows, d1_input)
    pose_root = out_dir / "d1_vggt"
    cache_root = pose_root / "cache"
    d1_scored = out_dir / "d1_scored.jsonl"
    d1_camalign_scored = out_dir / "d1_camalign_scored.jsonl"
    d1_camalign_summary = out_dir / "d1_camalign_summary.json"
    d1_summary = out_dir / "d1_summary.csv"
    d2_out = out_dir / "d2_features.jsonl"
    d3d6_out = out_dir / "d3d6"
    runtime_scores = d3d6_out / D3D6_SCORES

    rc = d1_vggt_batch(
        eval_runtime=eval_runtime,
        input_jsonl=d1_input,
        output_root=pose_root,
        cache_root=cache_root,
    )
    if rc != 0:
        return rc

    d1_defaults = DEFAULT_D1_SCORER_CONFIG
    rc = d1_score(
        input_jsonl=d1_input,
        output_jsonl=d1_scored,
        summary_csv=d1_summary,
        pose_cache_root=cache_root,
        pose_backend=_extra_or_default(
            eval_runtime.scorers, "d1_pose_backend", eval_runtime.backends.get("pose", d1_defaults["pose_backend"])
        ),
        poses_file=_extra_or_default(eval_runtime.scorers, "d1_poses_file", d1_defaults["poses_file"]),
        default_frames=_extra_int_or_default(eval_runtime.scorers, "d1_default_frames", d1_defaults["default_frames"]),
        sidecar_profile_gate=sidecar_profile_gate,
        target_fallback_mode=target_fallback_mode,
        predicted_pose_type=_extra_or_default(
            eval_runtime.scorers, "d1_predicted_pose_type", d1_defaults["predicted_pose_type"]
        ),
        predicted_camera_convention=_extra_or_default(
            eval_runtime.scorers, "d1_predicted_camera_convention", d1_defaults["predicted_camera_convention"]
        ),
        target_camera_convention=_extra_or_default(
            eval_runtime.scorers, "d1_target_camera_convention", d1_defaults["target_camera_convention"]
        ),
        rot_scale_deg=_extra_float_or_default(eval_runtime.scorers, "d1_rot_scale_deg", d1_defaults["rot_scale_deg"]),
        trans_scale=_extra_float_or_default(eval_runtime.scorers, "d1_trans_scale", d1_defaults["trans_scale"]),
        yaw_weak_threshold_deg=_extra_float_or_default(
            eval_runtime.scorers, "d1_yaw_weak_threshold_deg", d1_defaults["yaw_weak_threshold_deg"]
        ),
        pan_weak_threshold=_extra_float_or_default(
            eval_runtime.scorers, "d1_pan_weak_threshold", d1_defaults["pan_weak_threshold"]
        ),
        static_rot_threshold_deg=_extra_float_or_default(
            eval_runtime.scorers, "d1_static_rot_threshold_deg", d1_defaults["static_rot_threshold_deg"]
        ),
        static_trans_threshold=_extra_float_or_default(
            eval_runtime.scorers, "d1_static_trans_threshold", d1_defaults["static_trans_threshold"]
        ),
    )
    if rc != 0:
        return rc

    rc = d1_camalign_score(
        input_jsonl=d1_input,
        output_jsonl=d1_camalign_scored,
        pose_cache_root=cache_root,
        poses_file=_extra_or_default(eval_runtime.scorers, "d1_poses_file", d1_defaults["poses_file"]),
        scorer_version=camalign_scorer_version,
        summary_json=d1_camalign_summary,
    )
    if rc != 0:
        return rc

    rc = d2_extract(
        eval_runtime=eval_runtime,
        videos_manifest=manifest_json,
        out_jsonl=d2_out,
    )
    if rc != 0:
        return rc

    rc = d3d6_score(
        eval_runtime=eval_runtime,
        manifest=manifest_json,
        out_dir=d3d6_out,
        stage="all",
        scorer_profile=scorer_profile,
    )
    if rc != 0:
        return rc

    return build_table(
        [
            "--runtime-scores",
            str(runtime_scores),
            "--d1-scores",
            str(d1_scored),
            "--d1-camalign-scores",
            str(d1_camalign_scored),
            "--d2-scores",
            str(d2_out),
            "--out-csv",
            str(out_dir / "main_table.csv"),
            "--out-md",
            str(out_dir / "main_table.md"),
            "--out-summary",
            str(out_dir / "main_table_summary.json"),
        ]
    )


def build_table(argv: list[str] | None = None) -> int:
    from wrbench.eval.aggregate.build_main_table import main as table_main

    return table_main(argv)
