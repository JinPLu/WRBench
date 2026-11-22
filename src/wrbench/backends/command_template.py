"""Declarative command-template generation backend.

Executes any model inference script or wrapper CLI configured via
``ModelRuntime.command_template`` in ``wrbench.runtime.json`` without writing
a custom Python launcher.
"""

from __future__ import annotations

import os
import string
import subprocess
from pathlib import Path
from typing import Any

from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.registry import canonical_model_key
from wrbench.runtime import ModelRuntime, RuntimeConfig

_FORMATTER = string.Formatter()


def _format_token(token: str, values: dict[str, str]) -> str:
    parts: list[str] = []
    for literal_text, field_name, format_spec, conversion in _FORMATTER.parse(token):
        if literal_text:
            parts.append(literal_text)
        if field_name is None:
            continue
        if conversion or format_spec or "." in field_name or "[" in field_name or not field_name:
            raise ValueError(
                f"Unsafe or unsupported placeholder syntax in command_template token: {token!r}"
            )
        if field_name not in values:
            raise ValueError(
                f"Unknown command_template placeholder {{{field_name}}} "
                f"(available: {', '.join(sorted(values))})"
            )
        parts.append(values[field_name])
    return "".join(parts)


def _placeholder_values(request: GenerationRequest, runtime: ModelRuntime, *, key: str) -> dict[str, str]:
    output_path = Path(request.output_path).resolve()
    work_dir = Path(request.work_dir).resolve() if request.work_dir is not None else output_path.parent
    extra = request.extra if isinstance(request.extra, dict) else {}
    payload_dict = request.payload.payload if isinstance(request.payload.payload, dict) else {}

    values: dict[str, str] = {str(k): str(v) for k, v in runtime.extra_paths.items()}
    values.update(
        {
            "python_bin": str(runtime.python_bin),
            "repo_root": str(runtime.repo_root),
            "model_path": str(runtime.model_path or ""),
            "gpu_id": str(runtime.gpu_id) if runtime.gpu_id is not None else "",
            "model": key,
            "prompt": str(request.prompt),
            "prompt_to_send": str(payload_dict.get("prompt_to_send", request.prompt)),
            "output_path": str(output_path),
            "image_path": str(Path(request.image_path).resolve()) if request.image_path else "",
            "source_video_path": str(Path(request.source_video_path).resolve()) if request.source_video_path else "",
            "work_dir": str(work_dir),
            "payload_json_path": str(
                extra.get("payload_json_path") or output_path.with_suffix(output_path.suffix + ".payload.json")
            ),
            "target_c2w_path": str(
                extra.get("target_c2w_path") or output_path.with_suffix(output_path.suffix + ".target_c2w.npy")
            ),
            "camera_trajectory_json_path": str(
                extra.get("camera_trajectory_json_path")
                or output_path.with_suffix(output_path.suffix + ".camera_trajectory.json")
            ),
            "camera_sidecar_path": str(
                extra.get("camera_sidecar_path") or output_path.with_suffix(output_path.suffix + ".camera.json")
            ),
        }
    )
    if payload_dict.get("pose_file"):
        values.setdefault("pose_file", str(payload_dict["pose_file"]))
    if payload_dict.get("actions_json"):
        values.setdefault("actions_json", str(payload_dict["actions_json"]))
    if payload_dict.get("prompt_txt"):
        values.setdefault("prompt_txt", str(payload_dict["prompt_txt"]))
    return values


class CommandTemplateBackend:
    """Launch any model command configured via ``ModelRuntime.command_template``."""

    name = "command_template"

    def __init__(self, runtime: RuntimeConfig) -> None:
        self._runtime = runtime

    def available(self) -> tuple[bool, str]:
        configured = sorted(
            key for key, rt in self._runtime.models.items() if rt.command_template
        )
        if not configured:
            return False, "runtime config has no models with command_template configured"
        return True, f"command_template configured for {', '.join(configured)}"

    def available_for(self, model: str) -> tuple[bool, str]:
        key = canonical_model_key(model)
        rt = self._runtime.model(key)
        if rt is None:
            return False, f"no runtime entry for {key!r} in wrbench.runtime.json"
        if not rt.command_template:
            return False, f"no command_template configured for {key!r} in wrbench.runtime.json"
        return True, f"command_template ready for {key}"

    def build_command(self, request: GenerationRequest, runtime: ModelRuntime) -> list[str]:
        if not runtime.command_template:
            raise ValueError(f"{request.model}: command_template is not configured")
        key = canonical_model_key(request.model)
        values = _placeholder_values(request, runtime, key=key)
        return [_format_token(token, values) for token in runtime.command_template]

    def generate(self, request: GenerationRequest) -> GenerationResult:
        key = canonical_model_key(request.model)
        ok, reason = self.available_for(key)
        if not ok:
            return GenerationResult(success=False, message=reason)

        runtime = self._runtime.model(key)
        assert runtime is not None
        try:
            cmd = self.build_command(request, runtime)
        except ValueError as exc:
            return GenerationResult(success=False, message=str(exc))

        output_path = Path(request.output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        repo_path = Path(runtime.repo_root).expanduser() if runtime.repo_root else None
        if repo_path is not None and repo_path.is_dir():
            cwd = repo_path.resolve()
        else:
            cwd = (Path(request.work_dir).resolve() if request.work_dir else output_path.parent)

        env = dict(os.environ)
        if runtime.gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in runtime.env:
            env["CUDA_VISIBLE_DEVICES"] = str(runtime.gpu_id)
        env.update(runtime.env)

        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            tail = "\n".join(part for part in (proc.stdout, proc.stderr) if part)[-4096:]
            return GenerationResult(
                success=False,
                output_path=output_path if output_path.is_file() else None,
                message=f"command_template failed (exit {proc.returncode}): {tail}",
            )
        if not output_path.is_file():
            return GenerationResult(
                success=False,
                message=f"command_template exited 0 but output missing: {output_path}",
            )
        cmd_preview = " ".join(cmd[:8]) + (" ..." if len(cmd) > 8 else "")
        return GenerationResult(
            success=True,
            output_path=output_path,
            message="generation completed",
            artifacts={"output_mp4": str(output_path), "command": cmd_preview},
        )


__all__ = ["CommandTemplateBackend"]
