"""Build Spatia inference.py subprocess commands."""

from __future__ import annotations

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


def _require_extra(runtime: ModelRuntime, key: str) -> str:
    value = runtime.extra_paths.get(key)
    if value is None or not str(value).strip():
        raise ValueError(f"{runtime.key}: runtime.extra_paths.{key} is required")
    return str(value)


def _write_intrinsics_file(out_path: Path, intrinsics: Any) -> Path:
    arr = np.asarray(intrinsics, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    k = arr[0]
    intrinsics_path = out_path.with_suffix(out_path.suffix + ".payload.intrinsics.txt")
    intrinsics_path.write_text(
        f"{float(k[0, 0]):.8f} {float(k[1, 1]):.8f} {float(k[0, 2]):.8f} {float(k[1, 2]):.8f}\n",
        encoding="utf-8",
    )
    return intrinsics_path


def build_spatia_command(
    *,
    model: str,
    payload: dict[str, Any],
    runtime: ModelRuntime,
    source_video_path: Path,
    prompt: str,
    output_path: Path,
    width: int,
    height: int,
    max_frames: int,
    fps: int,
) -> tuple[list[str], Path, dict[str, str], Path]:
    if not runtime.repo_root:
        raise ValueError(f"{model}: runtime.repo_root (spatia_dir) is required")
    if not runtime.python_bin:
        raise ValueError(f"{model}: runtime.python_bin is required")

    w2c_path = payload.get("w2c_trajectory_file")
    if not w2c_path:
        raise ValueError(f"{model}: payload missing w2c_trajectory_file")

    vace_path = _require_extra(runtime, "vace_path")
    lora_path = _require_extra(runtime, "lora_path")
    num_inference_steps = _require_extra(runtime, "num_inference_steps")
    cfg_scale = _require_extra(runtime, "cfg_scale")
    sigma_shift = _require_extra(runtime, "sigma_shift")
    seed = _require_extra(runtime, "seed")
    ffmpeg_bin = _require_extra(runtime, "ffmpeg_bin")

    spatia_dir = Path(runtime.repo_root)
    inference_py = spatia_dir / "inference.py"
    if not inference_py.is_file():
        raise FileNotFoundError(f"Spatia inference.py not found: {inference_py}")

    frame_path = output_path.with_suffix(output_path.suffix + ".first_frame.png")
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    _extract_first_frame(source_video_path, frame_path, ffmpeg_bin=Path(ffmpeg_bin))

    intrinsics_path = _write_intrinsics_file(output_path, payload.get("intrinsics"))
    work_dir = Path(str(output_path.with_suffix("")) + "_assets")
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(runtime.python_bin),
        str(inference_py),
        "--img_path",
        str(frame_path),
        "--camera_w2c_path",
        str(w2c_path),
        "--camera_intrinsics_path",
        str(intrinsics_path),
        "--vace_path",
        str(vace_path),
        "--lora_path",
        str(lora_path),
        "--save_path",
        str(output_path),
        "--work_dir",
        str(work_dir),
        "--prompt",
        prompt,
        "--prompt_path",
        "",
        "--width",
        str(width),
        "--height",
        str(height),
        "--max_frames",
        str(max_frames),
        "--first_round_frames",
        str(max_frames),
        "--fps",
        str(fps),
        "--num_inference_steps",
        num_inference_steps,
        "--cfg_scale",
        cfg_scale,
        "--sigma_shift",
        sigma_shift,
        "--seed",
        seed,
    ]

    env = dict(runtime.env)
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    return cmd, spatia_dir, env, frame_path


def validate_spatia_runtime(runtime: ModelRuntime) -> list[str]:
    missing = []
    if not runtime.python_bin or not Path(str(runtime.python_bin)).is_file():
        missing.append("python_bin")
    if not runtime.repo_root or not Path(str(runtime.repo_root)).is_dir():
        missing.append("repo_root")
    for field in ("vace_path", "lora_path", "ffmpeg_bin"):
        if field not in runtime.extra_paths or not Path(str(runtime.extra_paths[field])).is_file():
            missing.append(f"extra_paths.{field}")
    for field in ("num_inference_steps", "cfg_scale", "sigma_shift", "seed"):
        if field not in runtime.extra_paths or not str(runtime.extra_paths[field]).strip():
            missing.append(f"extra_paths.{field}")
    return missing


def _prepare(request: GenerationRequest, runtime: ModelRuntime) -> PreparedLaunch:
    output_path = Path(request.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = model_record("spatia")
    cmd, cwd, env, _ = build_spatia_command(
        model="spatia",
        payload=dict(request.payload.payload),
        runtime=runtime,
        source_video_path=require_source_video(request, "spatia"),
        prompt=require_prompt(request, "spatia"),
        output_path=output_path,
        width=record.default_width,
        height=record.default_height,
        max_frames=record.default_frames,
        fps=record.default_fps,
    )
    return PreparedLaunch(cmd, cwd, env, lambda: successful_generation(output_path, cmd))


SPEC = LaunchSpec("spatia", validate_spatia_runtime, _prepare)


def _extract_first_frame(source_video: Path, out_png: Path, *, ffmpeg_bin: Path) -> None:
    import subprocess

    if not ffmpeg_bin.is_file():
        raise FileNotFoundError(f"Configured ffmpeg_bin not found: {ffmpeg_bin}")
    proc = subprocess.run(
        [
            str(ffmpeg_bin),
            "-y",
            "-i",
            str(source_video),
            "-frames:v",
            "1",
            str(out_png),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out_png.is_file():
        tail = "\n".join(part for part in (proc.stdout, proc.stderr) if part)[-2048:]
        raise RuntimeError(f"Configured ffmpeg failed to extract first frame: {tail}")
