"""Backend registry and resolution."""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from wrbench.backends.base import DryRunBackend, GenerationBackend
from wrbench.backends.command_template import CommandTemplateBackend
from wrbench.backends.local_subprocess import LocalSubprocessBackend
from wrbench.registry import canonical_model_key
from wrbench.runtime import RuntimeConfig, load_runtime_config

_BUILTIN_BACKEND_NAMES = ("dry_run", "local_subprocess", "command_template")
_GENERATION_BACKENDS: dict[str, Any] = {}


def register_generation_backend(name: str, factory: Any) -> Any:
    """Register a named generation backend factory, class, instance, or ``module:callable`` spec."""
    key = str(name or "").strip().lower()
    if not key:
        raise ValueError("Generation backend name must be a non-empty string")
    if not (
        callable(factory)
        or isinstance(factory, str)
        or (hasattr(factory, "generate") and callable(factory.generate))
    ):
        raise ValueError(
            "Generation backend must be a callable factory/class, backend instance, or 'module:callable' string"
        )
    _GENERATION_BACKENDS[key] = factory
    return factory


def registered_generation_backends() -> list[str]:
    """Return all available built-in and dynamically registered generation backend names."""
    return sorted(set(_BUILTIN_BACKEND_NAMES) | set(_GENERATION_BACKENDS))


def _load_import_target(spec: str) -> Any:
    module_name, sep, attr_name = spec.partition(":")
    module_name = module_name.strip()
    attr_name = attr_name.strip()
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid backend spec {spec!r}; expected 'module:callable'")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot load generation backend {spec!r}: {exc}") from exc


def _instantiate_backend(
    target: Any,
    *,
    runtime: RuntimeConfig | None,
    model_key: str,
) -> GenerationBackend:
    if isinstance(target, str):
        target = _load_import_target(target)
    if not isinstance(target, type) and hasattr(target, "generate") and callable(target.generate):
        return target
    if not callable(target):
        raise ValueError(f"Generation backend {target!r} is not callable and has no .generate()")

    sig = inspect.signature(target)
    params = sig.parameters
    has_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs: dict[str, Any] = {}
    if "runtime" in params or has_var_kw:
        kwargs["runtime"] = runtime
    if "model" in params:
        kwargs["model"] = model_key

    if kwargs:
        backend = target(**kwargs)
    else:
        required_pos = [
            p
            for p in params.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not required_pos:
            backend = target()
        elif len(required_pos) == 1:
            backend = target(runtime)
        else:
            raise ValueError(
                f"Generation backend factory {target!r} requires unsupported arguments: {sig}"
            )
    if not hasattr(backend, "generate") or not callable(backend.generate):
        raise ValueError(f"Generation backend factory {target!r} did not return a GenerationBackend")
    return backend


def resolve_backend(
    model: str,
    *,
    runtime: RuntimeConfig | None = None,
    backend_name: str | None = None,
) -> GenerationBackend:
    """Return the backend selected by explicit config, runtime entry, and model support."""

    key = canonical_model_key(model)
    runtime = runtime if runtime is not None else load_runtime_config()
    model_rt = runtime.model(key) if runtime is not None else None

    selected = str(backend_name).strip() if backend_name else ""
    if not selected and model_rt is not None and model_rt.backend:
        selected = str(model_rt.backend).strip()

    if not selected:
        if model_rt is not None:
            if model_rt.command_template:
                return CommandTemplateBackend(runtime)
            return LocalSubprocessBackend(runtime)
        return DryRunBackend()

    lower = selected.lower()
    if lower == "dry_run":
        return DryRunBackend()
    if lower == "local_subprocess":
        if runtime is None:
            raise RuntimeError("backend 'local_subprocess' requires an explicit wrbench.runtime.json")
        return LocalSubprocessBackend(runtime)
    if lower == "command_template":
        if runtime is None:
            raise RuntimeError("backend 'command_template' requires an explicit wrbench.runtime.json")
        return CommandTemplateBackend(runtime)
    if lower in _GENERATION_BACKENDS:
        return _instantiate_backend(_GENERATION_BACKENDS[lower], runtime=runtime, model_key=key)
    if ":" in selected:
        return _instantiate_backend(selected, runtime=runtime, model_key=key)
    raise ValueError(f"Unknown backend {selected!r}")


def list_backends(model: str, *, runtime: RuntimeConfig | None = None) -> list[tuple[str, bool, str]]:
    """Return ``(name, available, reason)`` tuples for *model*."""

    key = canonical_model_key(model)
    runtime = runtime if runtime is not None else load_runtime_config()
    rows: list[tuple[str, bool, str]] = []
    dry = DryRunBackend()
    ok, msg = dry.available()
    rows.append((dry.name, ok, msg))
    if runtime is not None:
        local = LocalSubprocessBackend(runtime)
        ok, msg = local.available_for(key)
        rows.append((local.name, ok, msg))
        tmpl = CommandTemplateBackend(runtime)
        ok, msg = tmpl.available_for(key)
        rows.append((tmpl.name, ok, msg))
    else:
        rows.append(("local_subprocess", False, "explicit wrbench.runtime.json not provided"))
        rows.append(("command_template", False, "explicit wrbench.runtime.json not provided"))

    seen = {name for name, _, _ in rows}
    for name in sorted(_GENERATION_BACKENDS):
        if name in seen:
            continue
        try:
            backend = _instantiate_backend(_GENERATION_BACKENDS[name], runtime=runtime, model_key=key)
            if hasattr(backend, "available_for"):
                ok, msg = backend.available_for(key)  # type: ignore[attr-defined]
            else:
                ok, msg = backend.available()
            rows.append((name, bool(ok), str(msg)))
        except Exception as exc:
            rows.append((name, False, str(exc)))
        seen.add(name)

    model_rt = runtime.model(key) if runtime is not None else None
    if model_rt is not None and model_rt.backend:
        configured = str(model_rt.backend).strip()
        if configured.lower() not in seen and configured not in seen:
            try:
                backend = resolve_backend(key, runtime=runtime, backend_name=configured)
                if hasattr(backend, "available_for"):
                    ok, msg = backend.available_for(key)  # type: ignore[attr-defined]
                else:
                    ok, msg = backend.available()
                rows.append((configured, bool(ok), str(msg)))
            except Exception as exc:
                rows.append((configured, False, str(exc)))
    return rows
