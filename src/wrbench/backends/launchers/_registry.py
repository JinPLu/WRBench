"""Small launcher contract used by the local subprocess backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from wrbench.backends.base import GenerationRequest, GenerationResult
from wrbench.runtime import ModelRuntime


@dataclass(frozen=True)
class PreparedLaunch:
    cmd: list[str]
    cwd: Path
    env: dict[str, str]
    finalize: Callable[[], GenerationResult]


@dataclass(frozen=True)
class LaunchSpec:
    key: str
    validate_runtime: Callable[[ModelRuntime], list[str]]
    prepare: Callable[[GenerationRequest, ModelRuntime], PreparedLaunch]


def require_image(request: GenerationRequest, model: str) -> Path:
    if not request.image_path:
        raise ValueError(f"{model} requires image_path")
    return Path(request.image_path).resolve()


def require_source_video(request: GenerationRequest, model: str) -> Path:
    if not request.source_video_path:
        raise ValueError(f"{model} requires source_video_path")
    return Path(request.source_video_path).resolve()


def require_prompt(request: GenerationRequest, model: str) -> str:
    prompt = str(request.prompt)
    if not prompt.strip():
        raise ValueError(f"{model} requires prompt")
    return prompt


def successful_generation(output_path: Path, cmd: list[str]) -> GenerationResult:
    if not output_path.is_file():
        return GenerationResult(success=False, message=f"subprocess exited 0 but output missing: {output_path}")
    return GenerationResult(
        success=True,
        output_path=output_path,
        message="generation completed",
        artifacts={"output_mp4": str(output_path), "command": " ".join(cmd[:8]) + " ..."},
    )
