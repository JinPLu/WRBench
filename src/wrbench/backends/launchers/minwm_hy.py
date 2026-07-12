"""Build minWM HY Action2V subprocess commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.backends.launchers._registry import LaunchSpec, PreparedLaunch, require_image, require_prompt, successful_generation
from wrbench.backends.launchers.minwm_common import (
    expected_output,
    prepare_output_dir,
    runtime_env,
    runtime_missing_fields,
    runtime_value,
    torchrun_bin,
    torchrun_command,
)
from wrbench.contracts import require_execution_contract, require_mapping, require_str
from wrbench.registry import model_record
from wrbench.runtime import ModelRuntime

_ENTRYPOINT = "HY15/hy15_inference.py"
_REQUIRED_EXTRA_PATHS = (
    "mode",
    "chunk_latent_frames",
    "num_inference_steps",
    "shift",
    "guidance_scale",
    "stabilization_level",
)
_EXISTING_EXTRA_PATHS = ("base_model_path", "torchrun_bin")


def minwm_hy_torchrun_bin(runtime: ModelRuntime) -> Path:
    """Resolve the explicitly configured torchrun executable for minWM HY."""
    return torchrun_bin(runtime)


def minwm_hy_torchrun_command(runtime: ModelRuntime) -> list[str]:
    """Return the explicitly configured torch distributed launcher command prefix."""
    return torchrun_command(runtime)


def minwm_hy_expected_output(output_dir: Path) -> Path | None:
    """Return the newest MP4 written by official HY inference, if present."""
    return expected_output(output_dir, recursive=True)


def validate_minwm_hy_runtime(runtime: ModelRuntime) -> list[str]:
    return runtime_missing_fields(
        runtime,
        model_path_kind="dir",
        entrypoint_rel=_ENTRYPOINT,
        required_extra_paths=_REQUIRED_EXTRA_PATHS,
        existing_extra_paths=_EXISTING_EXTRA_PATHS,
    )


def _load_request_json(payload: dict[str, Any]) -> dict[str, Any]:
    request_json = payload.get("request_json")
    if not request_json:
        return {}
    path = Path(str(request_json))
    if not path.is_file():
        raise FileNotFoundError(f"minWM HY request JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _token_details(payload: dict[str, Any]) -> dict[str, Any]:
    details = payload.get("token_mapping_details")
    if isinstance(details, dict):
        return dict(details)
    request = _load_request_json(payload)
    contract = request.get("input_contract") if isinstance(request.get("input_contract"), dict) else {}
    details = contract.get("token_mapping_details") if isinstance(contract, dict) else None
    if isinstance(details, dict):
        return dict(details)
    trajectory = payload.get("trajectory")
    if isinstance(trajectory, dict):
        maybe = trajectory.get("token_details")
        if isinstance(maybe, dict):
            return dict(maybe)
    return {}


def _materialize_example_json(
    *,
    payload: dict[str, Any],
    image_path: Path,
    prompt: str,
    work_dir: Path,
) -> Path:
    template_path = Path(str(payload.get("example_json") or ""))
    if not template_path.is_file():
        raise FileNotFoundError(f"minWM HY example JSON template not found: {template_path}")
    if not image_path.is_file():
        raise FileNotFoundError(f"minWM HY first-frame image not found: {image_path}")
    prompt_text = str(prompt).strip()
    if not prompt_text:
        raise ValueError("minwm-hy-action2v requires a non-empty prompt for real generation")

    rows = json.loads(template_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"minWM HY example JSON must be a non-empty list: {template_path}")
    materialized: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("minWM HY example JSON rows must be objects")
        item = dict(row)
        item["image"] = str(image_path.resolve())
        item["caption"] = prompt_text
        materialized.append(item)

    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "minwm_hy_action2v_example.materialized.json"
    out.write_text(json.dumps(materialized, indent=2, sort_keys=True), encoding="utf-8")
    return out


def _materialize_entrypoint(
    *,
    source_entrypoint: Path,
    work_dir: Path,
    runtime_yaw_deg_per_token: float,
) -> Path:
    """Copy HY entrypoint and patch the native yaw token step when needed."""
    text = source_entrypoint.read_text(encoding="utf-8")
    old = "_ROT_STEP = np.radians(3.0)"
    new = f"_ROT_STEP = np.radians({repr(float(runtime_yaw_deg_per_token))})"
    if old not in text:
        raise ValueError(f"minWM HY entrypoint does not contain expected rotation step literal: {source_entrypoint}")
    patched = text.replace(
        old,
        new + "  # patched by WRBench for target yaw amplitude",
        1,
    )
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / "hy15_inference_wrbench_rotstep.py"
    out.write_text(patched, encoding="utf-8")
    return out


def _float_runtime_value(runtime: ModelRuntime, key: str) -> float:
    return float(runtime_value(runtime, key))


def build_minwm_hy_command(
    *,
    model: str,
    payload: dict[str, Any],
    runtime: ModelRuntime,
    image_path: Path,
    prompt: str,
    output_path: Path,
) -> tuple[list[str], Path, dict[str, str], Path]:
    """Build the official minWM HY DMD camera inference command."""
    execution = require_execution_contract(model)
    record = model_record(model)

    if not runtime.repo_root:
        raise ValueError(f"{model}: runtime.repo_root is required")
    if not runtime.python_bin:
        raise ValueError(f"{model}: runtime.python_bin is required")

    repo = Path(runtime.repo_root)
    source_entrypoint = repo / require_str(execution, "entrypoint")
    if not source_entrypoint.is_file():
        raise FileNotFoundError(f"minWM HY entrypoint not found: {source_entrypoint}")

    launcher = minwm_hy_torchrun_bin(runtime)
    if not launcher.is_file():
        raise FileNotFoundError(f"minWM HY launcher not found: {launcher}")

    if not runtime.model_path:
        raise ValueError(f"{model}: runtime.model_path is required for HY transformer_dir")
    transformer_dir = Path(str(runtime.model_path))
    if not transformer_dir.exists():
        raise FileNotFoundError(
            "minWM HY transformer_dir not found; set runtime.model_path in wrbench.runtime.json: "
            f"{transformer_dir}"
        )
    base_model_path = Path(runtime_value(runtime, "base_model_path"))
    if not base_model_path.exists():
        raise FileNotFoundError(
            "minWM HY base model path not found; set runtime.extra_paths.base_model_path "
            f"in wrbench.runtime.json: {base_model_path}"
        )

    work_dir = Path(str(output_path.with_suffix("")) + "_minwm_hy_work")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = prepare_output_dir(output_path, "_minwm_hy_output")

    materialized_example = _materialize_example_json(
        payload=payload,
        image_path=image_path,
        prompt=prompt,
        work_dir=work_dir,
    )

    details = _token_details(payload)
    requires_patch = bool(details.get("requires_runtime_rot_step_patch"))
    entrypoint_arg = require_str(execution, "entrypoint")
    if requires_patch:
        if "runtime_yaw_deg_per_token" not in details:
            raise ValueError(f"{model}: token_mapping_details.runtime_yaw_deg_per_token is required")
        runtime_yaw = float(details["runtime_yaw_deg_per_token"])
        patched_entrypoint = _materialize_entrypoint(
            source_entrypoint=source_entrypoint,
            work_dir=work_dir,
            runtime_yaw_deg_per_token=runtime_yaw,
        )
        entrypoint_arg = str(patched_entrypoint)

    mode = runtime_value(runtime, "mode")
    chunk_latent_frames = int(runtime_value(runtime, "chunk_latent_frames"))
    num_inference_steps = int(runtime_value(runtime, "num_inference_steps"))
    shift = _float_runtime_value(runtime, "shift")
    guidance_scale = _float_runtime_value(runtime, "guidance_scale")
    stabilization_level = int(runtime_value(runtime, "stabilization_level"))

    cmd = [
        *minwm_hy_torchrun_command(runtime),
        "--standalone",
        "--nproc_per_node=1",
        entrypoint_arg,
        "--use_camera",
        "--mode",
        mode,
        "--transformer_dir",
        str(transformer_dir),
        "--model_path",
        str(base_model_path),
        "--example_json",
        str(materialized_example),
        "--output_dir",
        str(output_dir),
        "--num_inference_steps",
        str(num_inference_steps),
        "--shift",
        str(shift),
        "--guidance_scale",
        str(guidance_scale),
        "--fps",
        str(int(record.default_fps)),
        "--height",
        str(int(record.default_height)),
        "--width",
        str(int(record.default_width)),
        "--video_length",
        str(int(record.default_frames)),
        "--chunk_latent_frames",
        str(chunk_latent_frames),
        "--stabilization_level",
        str(stabilization_level),
    ]

    env = runtime_env(runtime, pythonpath_entries=[repo / "HY15", repo / "shared", repo])
    return cmd, repo, env, output_dir


def _prepare(request: GenerationRequest, runtime: ModelRuntime) -> PreparedLaunch:
    output_path = Path(request.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, cwd, env, output_dir = build_minwm_hy_command(
        model="minwm-hy-action2v",
        payload=dict(request.payload.payload),
        runtime=runtime,
        image_path=require_image(request, "minwm-hy-action2v"),
        prompt=require_prompt(request, "minwm-hy-action2v"),
        output_path=output_path,
    )

    def finalize() -> GenerationResult:
        produced = minwm_hy_expected_output(output_dir)
        if produced is None:
            return GenerationResult(success=False, message=f"subprocess exited 0 but no minWM HY mp4 found under: {output_dir}")
        shutil.copy2(produced, output_path)
        return successful_generation(output_path, cmd)

    return PreparedLaunch(cmd, cwd, env, finalize)


SPEC = LaunchSpec("minwm-hy-action2v", validate_minwm_hy_runtime, _prepare)
