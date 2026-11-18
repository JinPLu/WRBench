"""Build official ReCamMaster subprocess commands."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from wrbench.backends.base import GenerationRequest
from wrbench.backends.launchers._registry import (
    LaunchSpec,
    PreparedLaunch,
    require_prompt,
    require_source_video,
    successful_generation,
)
from wrbench.registry import model_record
from wrbench.runtime import ModelRuntime


def _path(value: str) -> Path:
    # Keep a virtual environment's Python symlink so it retains its prefix.
    return Path(value).expanduser().absolute()


def validate_recammaster_runtime(runtime: ModelRuntime) -> list[str]:
    missing: list[str] = []
    python = _path(runtime.python_bin) if runtime.python_bin else None
    if python is None or not python.is_file() or not os.access(python, os.X_OK):
        missing.append("python_bin")
    repo = _path(runtime.repo_root) if runtime.repo_root else None
    if repo is None or not (repo / "diffsynth" / "pipelines" / "wan_video_recammaster.py").is_file():
        missing.append("repo_root")
    if not runtime.model_path or not _path(str(runtime.model_path)).is_file():
        missing.append("model_path")
    extra = runtime.extra_paths
    wan_dir = _path(extra["wan_model_dir"]) if extra.get("wan_model_dir") else None
    for name in (
        "diffusion_pytorch_model.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
    ):
        if wan_dir is None or not (wan_dir / name).is_file():
            missing.append(f"extra_paths.wan_model_dir ({name})")
    if wan_dir is None or not (wan_dir / "google" / "umt5-xxl").is_dir():
        missing.append("extra_paths.wan_model_dir (google/umt5-xxl tokenizer)")
    if "negative_prompt" not in extra:
        missing.append("extra_paths.negative_prompt")
    for field, parse, valid in (
        ("cfg_scale", float, lambda value: math.isfinite(value) and value > 0),
        ("num_inference_steps", int, lambda value: value > 0),
        ("seed", int, lambda value: 0 <= value < 2**64),
        ("tiled", str, lambda value: value.lower() in {"true", "false", "yes", "no", "1", "0"}),
    ):
        try:
            ok = valid(parse(extra.get(field, "").strip()))
        except ValueError:
            ok = False
        if not ok:
            missing.append(f"extra_paths.{field}")
    return missing


def build_recammaster_command(
    *,
    model: str,
    payload: dict[str, Any],
    runtime: ModelRuntime,
    source_video_path: Path,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
) -> tuple[list[str], Path, dict[str, str], Path]:
    missing = validate_recammaster_runtime(runtime)
    if missing:
        raise ValueError(f"{model}: missing or invalid runtime fields: {', '.join(missing)}")
    record = model_record("recammaster")
    required = record.default_width, record.default_height, record.default_frames
    if (width, height, num_frames) != required:
        raise ValueError(f"{model} generation requires {required[0]}x{required[1]} and {required[2]} frames")
    if not source_video_path.is_file():
        raise ValueError(f"{model}: source video not found: {source_video_path}")
    if fps <= 0 or output_path.suffix.lower() != ".mp4":
        raise ValueError(f"{model} requires a positive fps and an .mp4 output path")
    if source_video_path.resolve() == output_path.resolve():
        raise ValueError(f"{model}: source video and output must be different files")
    embedding = np.asarray(payload.get("camera_embedding"), dtype=np.float32)
    if embedding.shape != (1, 21, 12) or not np.isfinite(embedding).all():
        raise ValueError(f"{model}: camera_embedding must contain 21 finite relative poses with shape (1, 21, 12)")

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_path = output_path.with_suffix(output_path.suffix + ".camera_embedding.npy")
    np.save(camera_path, embedding, allow_pickle=False)
    runner = Path(__file__).resolve().parents[1] / "runners" / "recammaster.py"
    repo = _path(runtime.repo_root)
    extra = runtime.extra_paths
    cmd = [
        str(_path(runtime.python_bin)), str(runner),
        "--repo-root", str(repo),
        "--wan-model-dir", str(_path(extra["wan_model_dir"])),
        "--checkpoint", str(_path(str(runtime.model_path))),
        "--source-video", str(source_video_path.resolve()),
        "--camera-npy", str(camera_path),
        f"--prompt={prompt}",
        f"--negative-prompt={extra['negative_prompt']}",
        "--output", str(output_path),
        "--height", str(height), "--width", str(width),
        "--num-frames", str(num_frames), "--fps", str(fps),
        "--cfg-scale", extra["cfg_scale"],
        "--num-inference-steps", extra["num_inference_steps"],
        "--seed", extra["seed"],
        "--tiled", extra["tiled"],
    ]
    env = dict(runtime.env)
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    return cmd, repo, env, camera_path


def _prepare(request: GenerationRequest, runtime: ModelRuntime) -> PreparedLaunch:
    record = model_record("recammaster")
    output_path = Path(request.output_path).resolve()
    width, height = request.payload.metadata.get("output_image_size", [record.default_width, record.default_height])
    cmd, cwd, env, camera_path = build_recammaster_command(
        model="recammaster",
        payload=dict(request.payload.payload),
        runtime=runtime,
        source_video_path=require_source_video(request, "recammaster"),
        prompt=require_prompt(request, "recammaster"),
        output_path=output_path,
        width=width,
        height=height,
        num_frames=request.payload.metadata.get("output_frame_count", record.default_frames),
        fps=request.payload.target_trajectory.fps,
    )

    def finalize():
        result = successful_generation(output_path, cmd)
        if result.success:
            result.artifacts["camera_embedding"] = str(camera_path)
        return result

    return PreparedLaunch(cmd, cwd, env, finalize)


SPEC = LaunchSpec("recammaster", validate_recammaster_runtime, _prepare)
