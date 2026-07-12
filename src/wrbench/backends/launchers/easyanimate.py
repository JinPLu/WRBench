"""Build EasyAnimate V5.1 control-camera subprocess commands."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from wrbench.backends.launchers._registry import (
    LaunchSpec,
    PreparedLaunch,
    require_image,
    require_prompt,
    successful_generation,
)
from wrbench.contracts import require_bool, require_execution_contract, require_int, require_mapping, require_sequence, require_str
from wrbench.runtime import ModelRuntime


def _replace_assignment(text: str, name: str, value: str) -> str:
    pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)
    if not pattern.search(text):
        raise RuntimeError(f"EasyAnimate source does not contain assignment: {name}")
    return pattern.sub(f"{name.ljust(24)}= {value}", text, count=1)


def _require_float(node: dict[str, Any], field: str) -> float:
    value = node.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Missing required numeric EasyAnimate contract field: {field}")
    return float(value)


def _require_hw_pair(node: dict[str, Any], field: str) -> list[int]:
    value = require_sequence(node, field)
    if len(value) != 2:
        raise ValueError(f"EasyAnimate {field} must contain [height, width], got {value!r}")
    return [int(value[0]), int(value[1])]


def materialize_easyanimate_script(
    *,
    source: Path,
    localized: Path,
    model_path: str,
    control_camera_txt: Path,
    image_path: Path,
    prompt: str,
    save_dir: Path,
    runtime_parameters: dict[str, Any],
    official_defaults: dict[str, Any],
    official_profile: dict[str, Any],
    benchmark_profile: dict[str, Any],
    sample_size: list[int],
    video_length: int,
    fps: int,
    enable_teacache: bool | None = None,
) -> Path:
    """Patch ``predict_v2v_control.py`` defaults (WRBench-style materialization)."""
    text = source.read_text(encoding="utf-8")
    if enable_teacache is None:
        enable_teacache = require_bool(official_profile, "enable_teacache")
    replacements = {
        "GPU_memory_mode": json.dumps(require_str(runtime_parameters, "gpu_memory_mode")),
        "enable_teacache": "True" if enable_teacache else "False",
        "teacache_threshold": str(_require_float(official_profile, "teacache_threshold")),
        "config_path": json.dumps(require_str(runtime_parameters, "config_path")),
        "model_name": json.dumps(model_path),
        "sampler_name": json.dumps(require_str(official_profile, "sampler_name")),
        "sample_size": json.dumps([int(sample_size[0]), int(sample_size[1])]),
        "video_length": str(int(video_length)),
        "fps": str(int(fps)),
        "weight_dtype": require_str(runtime_parameters, "weight_dtype"),
        "control_video": "None",
        "control_camera_txt": json.dumps(str(control_camera_txt)),
        "ref_image": json.dumps(str(image_path)),
        "prompt": json.dumps(prompt),
        "guidance_scale": str(_require_float(official_profile, "guidance_scale")),
        "seed": str(require_int(official_profile, "seed")),
        "num_inference_steps": str(require_int(official_profile, "num_inference_steps")),
        "save_path": json.dumps(str(save_dir)),
    }
    for name, value in replacements.items():
        text = _replace_assignment(text, name, value)

    text = text.replace(
        '            if hasattr(_text_encoder, "visual"):\n                del _text_encoder.visual',
        '            try:\n                del _text_encoder.visual\n            except AttributeError:\n                pass',
    )
    text = text.replace(
        '        if hasattr(_text_encoder, "visual"):\n            del _text_encoder.visual',
        '        try:\n            del _text_encoder.visual\n        except AttributeError:\n            pass',
    )
    if require_bool(runtime_parameters, "manual_cuda_offload_patch"):
        text = text.replace(
            '    convert_weight_dtype_wrapper(transformer, weight_dtype)\n    pipeline.enable_model_cpu_offload()\nelse:\n    pipeline.enable_model_cpu_offload()',
            '    convert_weight_dtype_wrapper(transformer, weight_dtype)\n    pipeline.to("cuda", silence_dtype_warnings=True)\n    if isinstance(pipeline.text_encoder, Qwen2VLForConditionalGeneration):\n        pipeline.text_encoder.to("cpu")\n    if isinstance(pipeline.text_encoder_2, Qwen2VLForConditionalGeneration) and pipeline.text_encoder_2 is not None:\n        pipeline.text_encoder_2.to("cpu")\n    pipeline.manual_cpu_offload_flag = False\nelse:\n    pipeline.to("cuda", silence_dtype_warnings=True)\n    if isinstance(pipeline.text_encoder, Qwen2VLForConditionalGeneration):\n        pipeline.text_encoder.to("cpu")\n    if isinstance(pipeline.text_encoder_2, Qwen2VLForConditionalGeneration) and pipeline.text_encoder_2 is not None:\n        pipeline.text_encoder_2.to("cpu")\n    pipeline.manual_cpu_offload_flag = False',
        )
        text = text.replace(
            "\ncoefficients = get_teacache_coefficients(model_name)",
            '\ntype(pipeline)._execution_device = property(lambda self: torch.device("cuda"))\ncoefficients = get_teacache_coefficients(model_name)',
        )

    localized.parent.mkdir(parents=True, exist_ok=True)
    localized.write_text(text, encoding="utf-8")
    return localized


def easyanimate_expected_output(save_dir: Path) -> Path:
    """EasyAnimate writes indexed ``*.mp4`` files under ``save_path``."""
    mp4s = sorted(save_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    if mp4s:
        return mp4s[0]
    preferred = save_dir / "00000001.mp4"
    return preferred


def validate_easyanimate_runtime(runtime: ModelRuntime) -> list[str]:
    missing = []
    if not runtime.python_bin or not Path(str(runtime.python_bin)).is_file():
        missing.append("python_bin")
    if not runtime.repo_root or not Path(str(runtime.repo_root)).is_dir():
        missing.append("repo_root")
    if not runtime.model_path or not Path(str(runtime.model_path)).exists():
        missing.append("model_path")
    return missing


def build_easyanimate_command(
    *,
    model: str,
    payload: dict[str, Any],
    runtime: ModelRuntime,
    image_path: Path,
    prompt: str,
    output_path: Path,
) -> tuple[list[str], Path, dict[str, str], Path]:
    execution = require_execution_contract(model)
    runtime_parameters = dict(require_mapping(execution, "runtime_parameters"))
    official_defaults = dict(require_mapping(execution, "official_script_defaults"))
    benchmark_profile = dict(require_mapping(execution, "wrbench_benchmark_profile"))
    official_profile = dict(require_mapping(execution, "official_inference_profile"))

    if not runtime.repo_root:
        raise ValueError(f"{model}: runtime.repo_root is required")
    if not runtime.python_bin:
        raise ValueError(f"{model}: runtime.python_bin is required")
    if not runtime.model_path:
        raise ValueError(f"{model}: runtime.model_path is required")

    control_camera_txt = payload.get("control_camera_txt")
    if not control_camera_txt:
        raise ValueError(f"{model}: payload missing control_camera_txt")

    sample_size = _require_hw_pair(payload, "sample_size")
    height, width = int(sample_size[0]), int(sample_size[1])
    video_length = require_int(payload, "video_length")
    fps = require_int(payload, "fps")

    repo = Path(runtime.repo_root)
    entrypoint = require_str(execution, "entrypoint")
    script = repo / entrypoint
    if not script.is_file():
        raise FileNotFoundError(f"EasyAnimate entrypoint not found: {script}")

    output_path = output_path.resolve()
    save_dir = output_path.parent
    save_dir.mkdir(parents=True, exist_ok=True)
    localized = save_dir / f"{output_path.stem}_easyanimate_run.py"
    teacache_override: bool | None = None
    if "enable_teacache" in runtime.extra_paths:
        teacache_override = str(runtime.extra_paths["enable_teacache"]).lower() in {"1", "true", "yes"}
    materialize_easyanimate_script(
        source=script,
        localized=localized,
        model_path=str(runtime.model_path),
        control_camera_txt=Path(str(control_camera_txt)),
        image_path=image_path.resolve(),
        prompt=prompt,
        save_dir=save_dir,
        runtime_parameters=runtime_parameters,
        official_defaults=official_defaults,
        official_profile=official_profile,
        benchmark_profile=benchmark_profile,
        sample_size=[height, width],
        video_length=video_length,
        fps=fps,
        enable_teacache=teacache_override,
    )

    cmd = [str(runtime.python_bin), str(localized)]
    env = dict(runtime.env)
    env["CUDA_VISIBLE_DEVICES"] = str(runtime.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    repo_str = str(repo)
    if "PYTHONPATH" in env and env["PYTHONPATH"]:
        env["PYTHONPATH"] = f"{repo_str}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = repo_str
    return cmd, repo, env, easyanimate_expected_output(save_dir)


def _prepare(request: GenerationRequest, runtime: ModelRuntime) -> PreparedLaunch:
    output_path = Path(request.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, cwd, env, _ = build_easyanimate_command(
        model="easyanimate-v51-camera",
        payload=dict(request.payload.payload),
        runtime=runtime,
        image_path=require_image(request, "easyanimate-v51-camera"),
        prompt=require_prompt(request, "easyanimate-v51-camera"),
        output_path=output_path,
    )

    def finalize() -> GenerationResult:
        produced = easyanimate_expected_output(output_path.parent)
        if produced.is_file() and produced != output_path:
            if output_path.is_file():
                output_path.unlink()
            produced.replace(output_path)
        return successful_generation(output_path, cmd)

    return PreparedLaunch(cmd, cwd, env, finalize)


SPEC = LaunchSpec("easyanimate-v51-camera", validate_easyanimate_runtime, _prepare)
