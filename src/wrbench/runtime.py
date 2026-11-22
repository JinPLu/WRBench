"""Load local runtime paths for real generation backends."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelRuntime:
    """Per-model execution paths resolved from ``wrbench.runtime.json``."""

    key: str
    python_bin: str = sys.executable
    repo_root: str = "."
    gpu_id: int = 0
    model_path: str | None = None
    extra_paths: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    backend: str | None = None
    command_template: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    schema_version: int
    models: dict[str, ModelRuntime]
    model_spec_files: tuple[str, ...] = ()

    def model(self, key: str) -> ModelRuntime | None:
        return self.models.get(key)


class RuntimeConfigError(ValueError):
    """Raised when ``wrbench.runtime.json`` is present but invalid."""


def _resolve_runtime_path(explicit: Path | None) -> Path | None:
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"runtime config not found: {path}")
        return path
    return None


def _load_runtime_payload(resolved: Path) -> dict[str, Any]:
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeConfigError(f"{resolved}: runtime config root must be a JSON object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise RuntimeConfigError(f"{resolved}: required integer field 'schema_version' is missing or invalid")
    return payload


def _require_mapping(node: dict[str, Any], field: str, *, context: str) -> dict[str, Any]:
    value = node.get(field)
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{context}: required object field '{field}' is missing or invalid")
    return value


def _optional_mapping(node: dict[str, Any], field: str, *, context: str) -> dict[str, Any]:
    value = node.get(field)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeConfigError(f"{context}: optional object field '{field}' must be an object when present")
    return value


def _require_str(node: dict[str, Any], field: str, *, context: str) -> str:
    value = node.get(field)
    if not isinstance(value, str) or not value:
        raise RuntimeConfigError(f"{context}: required string field '{field}' is missing or invalid")
    return value


def _require_int(node: dict[str, Any], field: str, *, context: str) -> int:
    value = node.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeConfigError(f"{context}: required integer field '{field}' is missing or invalid")
    return int(value)


def _optional_str(node: dict[str, Any], field: str, *, context: str) -> str | None:
    value = node.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeConfigError(f"{context}: optional string field '{field}' must be a non-empty string")
    return value.strip()


def _optional_command_template(node: dict[str, Any], *, context: str) -> tuple[str, ...] | None:
    value = node.get("command_template")
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or not value or not all(isinstance(part, str) for part in value):
        raise RuntimeConfigError(f"{context}: optional field 'command_template' must be a non-empty list of strings")
    return tuple(str(part) for part in value)


def _load_model_spec_files(payload: dict[str, Any], *, resolved: Path) -> tuple[str, ...]:
    raw_specs = payload.get("model_spec_files")
    if raw_specs is None:
        return ()
    if not isinstance(raw_specs, list) or not all(isinstance(item, str) and item.strip() for item in raw_specs):
        raise RuntimeConfigError(f"{resolved}: optional field 'model_spec_files' must be a list of file path strings")
    from wrbench.registry import load_model_spec_file

    loaded: list[str] = []
    for item in raw_specs:
        candidate = Path(item).expanduser()
        if not candidate.is_absolute():
            candidate = (resolved.parent / candidate).resolve()
        load_model_spec_file(candidate, overwrite=True)
        loaded.append(str(candidate))
    return tuple(loaded)


def load_runtime_config(path: Path | None = None) -> RuntimeConfig | None:
    resolved = _resolve_runtime_path(path)
    if resolved is None:
        return None
    payload = _load_runtime_payload(resolved)
    spec_files = _load_model_spec_files(payload, resolved=resolved)

    if "models" in payload:
        model_nodes = _require_mapping(payload, "models", context=str(resolved))
    elif "eval" in payload or spec_files:
        model_nodes = {}
    else:
        model_nodes = _require_mapping(payload, "models", context=str(resolved))

    models: dict[str, ModelRuntime] = {}
    for key, node in sorted(model_nodes.items()):
        if not isinstance(node, dict):
            raise RuntimeConfigError(f"{resolved}: models.{key} must be an object")
        context = f"{resolved}: models.{key}"
        backend = _optional_str(node, "backend", context=context)
        command_template = _optional_command_template(node, context=context)
        allow_defaults = (backend is not None and backend != "local_subprocess") or (
            backend is None and command_template is not None
        )
        if allow_defaults:
            python_bin = (
                _require_str(node, "python_bin", context=context)
                if node.get("python_bin") is not None
                else sys.executable
            )
            repo_root = (
                _require_str(node, "repo_root", context=context)
                if node.get("repo_root") is not None
                else "."
            )
            gpu_id = (
                _require_int(node, "gpu_id", context=context)
                if node.get("gpu_id") is not None
                else 0
            )
        else:
            python_bin = _require_str(node, "python_bin", context=context)
            repo_root = _require_str(node, "repo_root", context=context)
            gpu_id = _require_int(node, "gpu_id", context=context)
        models[str(key)] = ModelRuntime(
            key=str(key),
            python_bin=python_bin,
            repo_root=repo_root,
            model_path=node.get("model_path"),
            gpu_id=gpu_id,
            extra_paths={str(k): str(v) for k, v in _optional_mapping(node, "extra_paths", context=context).items()},
            env={str(k): str(v) for k, v in _optional_mapping(node, "env", context=context).items()},
            backend=backend,
            command_template=command_template,
        )
    return RuntimeConfig(
        schema_version=int(payload["schema_version"]),
        models=models,
        model_spec_files=spec_files,
    )
