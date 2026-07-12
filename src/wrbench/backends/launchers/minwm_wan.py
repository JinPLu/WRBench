"""Build minWM Wan Action2V subprocess commands."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.backends.launchers._registry import LaunchSpec, PreparedLaunch, successful_generation
from wrbench.backends.launchers.minwm_common import (
    expected_output,
    prepare_output_dir,
    runtime_env,
    runtime_missing_fields,
    runtime_value,
    torchrun_bin,
    torchrun_command,
)
from wrbench.contracts import require_execution_contract, require_str
from wrbench.runtime import ModelRuntime

_ENTRYPOINT = "Wan21/wan_inference.py"
_REQUIRED_EXTRA_PATHS = ("num_output_frames", "sp_size")
_EXISTING_EXTRA_PATHS = ("config_path", "torchrun_bin")


def minwm_wan_torchrun_bin(runtime: ModelRuntime) -> Path:
    """Resolve the launcher executable used for minWM Wan.

    The launcher is intentionally configured explicitly because cluster
    wrappers and virtualenv entrypoints can point at different interpreters.
    """
    return torchrun_bin(runtime)


def minwm_wan_torchrun_command(runtime: ModelRuntime) -> list[str]:
    """Return the torch distributed launcher command prefix."""
    return torchrun_command(runtime)


def minwm_wan_expected_output(output_dir: Path) -> Path | None:
    """Return the newest MP4 written by official Wan inference, if present."""
    return expected_output(output_dir, recursive=False)


def validate_minwm_wan_runtime(runtime: ModelRuntime) -> list[str]:
    return runtime_missing_fields(
        runtime,
        model_path_kind="file",
        entrypoint_rel=_ENTRYPOINT,
        required_extra_paths=_REQUIRED_EXTRA_PATHS,
        existing_extra_paths=_EXISTING_EXTRA_PATHS,
    )


def build_minwm_wan_command(
    *,
    model: str,
    payload: dict[str, Any],
    runtime: ModelRuntime,
    prompt: str,
    output_path: Path,
) -> tuple[list[str], Path, dict[str, str], Path]:
    """Build the official minWM Wan DMD camera inference command."""
    execution = require_execution_contract(model)
    if not runtime.repo_root:
        raise ValueError(f"{model}: runtime.repo_root is required")
    if not runtime.python_bin:
        raise ValueError(f"{model}: runtime.python_bin is required")
    if not runtime.model_path:
        raise ValueError(f"{model}: runtime.model_path is required")

    repo = Path(runtime.repo_root)
    entrypoint = repo / require_str(execution, "entrypoint")
    if not entrypoint.is_file():
        raise FileNotFoundError(f"minWM Wan entrypoint not found: {entrypoint}")

    torchrun = minwm_wan_torchrun_bin(runtime)
    if not torchrun.is_file():
        raise FileNotFoundError(f"minWM Wan launcher not found: {torchrun}")

    prompt_txt = payload.get("prompt_txt")
    if not prompt_txt:
        raise ValueError(f"{model}: payload missing prompt_txt")
    prompt_path = Path(str(prompt_txt))
    if not prompt_path.is_file():
        raise FileNotFoundError(f"minWM Wan prompt file not found: {prompt_path}")

    trajectory_txt = payload.get("trajectory_txt")
    if not trajectory_txt:
        raise ValueError(f"{model}: payload missing trajectory_txt")
    trajectory_path = Path(str(trajectory_txt))
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"minWM Wan trajectory file not found: {trajectory_path}")

    output_dir = prepare_output_dir(output_path, "_minwm_wan_output")

    config_path = runtime_value(runtime, "config_path")
    num_output_frames = int(runtime_value(runtime, "num_output_frames"))
    sp_size = int(runtime_value(runtime, "sp_size"))

    cmd = [
        *minwm_wan_torchrun_command(runtime),
        "--standalone",
        "--nproc_per_node=1",
        require_str(execution, "entrypoint"),
        "--config_path",
        config_path,
        "--checkpoint_path",
        str(runtime.model_path),
        "--data_path",
        str(prompt_path),
        "--output_folder",
        str(output_dir),
        "--sp_size",
        str(sp_size),
        "--trajectory_path",
        str(trajectory_path),
        "--num_output_frames",
        str(num_output_frames),
    ]

    patch = payload.get("rotation_step_patch") or {}
    launcher = patch.get("launcher_path")
    if launcher:
        launcher_path = Path(str(launcher))
        if not launcher_path.is_file():
            raise FileNotFoundError(f"minWM Wan launcher patch not found: {launcher_path}")
        cmd = [str(launcher_path), *cmd]

    env = runtime_env(runtime, pythonpath_entries=[repo])
    if prompt and prompt_path.read_text(encoding="utf-8").strip() != prompt.strip():
        env["WRBENCH_PROMPT_MISMATCH_WARNING"] = "1"
    return cmd, repo, env, output_dir


def _prepare(request: GenerationRequest, runtime: ModelRuntime) -> PreparedLaunch:
    output_path = Path(request.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd, cwd, env, output_dir = build_minwm_wan_command(
        model="minwm-wan-action2v",
        payload=dict(request.payload.payload),
        runtime=runtime,
        prompt=request.prompt,
        output_path=output_path,
    )

    def finalize() -> GenerationResult:
        produced = minwm_wan_expected_output(output_dir)
        if produced is None:
            return GenerationResult(success=False, message=f"subprocess exited 0 but no minWM Wan mp4 found under: {output_dir}")
        shutil.copy2(produced, output_path)
        return successful_generation(output_path, cmd)

    return PreparedLaunch(cmd, cwd, env, finalize)


SPEC = LaunchSpec("minwm-wan-action2v", validate_minwm_wan_runtime, _prepare)
