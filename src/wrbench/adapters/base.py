"""Unified camera adapter protocol and registry.

An adapter turns a model-agnostic OpenCV ``CameraTrajectory`` into a model-native
``CameraPayload``. Adapters self-register with ``@register(...)`` against one or
more canonical model keys; ``wrbench.adapters`` imports every adapter module on
import so the registry is fully populated.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any, Callable, Protocol

from wrbench.payload import CameraPayload
from wrbench.registry import all_records, canonical_model_key, is_deferred_model, model_record
from wrbench.trajectory import CameraTrajectory


class CameraAdapter(Protocol):
    name: str

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
    ) -> CameraPayload:
        ...


class _CallableAdapter:
    """Wrap a plain compile function imported via ``module.path:fn``."""

    def __init__(self, name: str, fn: Callable[..., CameraPayload]) -> None:
        self.name = name
        self._fn = fn
        self._accepts_prompt = "prompt" in inspect.signature(fn).parameters

    def compile(
        self,
        trajectory: CameraTrajectory,
        *,
        model_name: str,
        width: int,
        height: int,
        num_frames: int,
        work_dir: str | Path | None = None,
        device: str | None = None,
        prompt: str = "",
    ) -> CameraPayload:
        kwargs: dict[str, Any] = {
            "model_name": model_name,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "work_dir": work_dir,
            "device": device,
        }
        if self._accepts_prompt:
            kwargs["prompt"] = prompt
        return self._fn(trajectory, **kwargs)


_REGISTRY: dict[str, CameraAdapter] = {}
_NAMED_ADAPTERS: dict[str, CameraAdapter] = {}
_DYNAMIC_ADAPTER_SPECS: dict[str, str] = {}


def register_named_adapter(name: str, adapter: CameraAdapter) -> None:
    """Register a reusable adapter by its symbolic adapter name (e.g. ``declarative_pose``)."""
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("Named adapter requires a non-empty name")
    _NAMED_ADAPTERS[cleaned] = adapter


def registered_named_adapters() -> list[str]:
    """List registered reusable/declarative adapter names."""
    return sorted(_NAMED_ADAPTERS)


def register_adapter(model_name_or_group: str | list[str] | tuple[str, ...], adapter: CameraAdapter) -> None:
    names = [model_name_or_group] if isinstance(model_name_or_group, str) else list(model_name_or_group)
    for name in names:
        key = canonical_model_key(name)
        if is_deferred_model(key):
            raise ValueError(f"Cannot register deferred model {key}")
        _REGISTRY[key] = adapter
        _DYNAMIC_ADAPTER_SPECS.pop(key, None)


def register(*model_keys: str) -> Callable[[type], type]:
    """Class decorator: instantiate the adapter and register it for the given keys."""

    def decorator(cls: type) -> type:
        register_adapter(list(model_keys), cls())
        return cls

    return decorator


def _looks_like_import_spec(spec: str) -> bool:
    module_name, sep, attr_name = spec.partition(":")
    return bool(sep and module_name.strip() and attr_name.strip())


def _resolve_import_adapter(spec: str) -> CameraAdapter:
    module_name, sep, attr_name = spec.partition(":")
    module_name = module_name.strip()
    attr_name = attr_name.strip()
    if not sep or not module_name or not attr_name:
        raise ValueError(f"Invalid adapter import spec {spec!r}; expected 'module.path:AttrName'")
    module = importlib.import_module(module_name)
    attr = getattr(module, attr_name)
    if isinstance(attr, type):
        instance = attr()
        if not hasattr(instance, "compile") or not callable(instance.compile):
            raise TypeError(f"Adapter class {spec!r} must implement a callable .compile(...) method")
        return instance
    if hasattr(attr, "compile") and callable(attr.compile):
        return attr
    if callable(attr):
        sig = inspect.signature(attr)
        required_positional = [
            p
            for p in sig.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if not required_positional:
            produced = attr()
            if hasattr(produced, "compile") and callable(produced.compile):
                return produced
            raise TypeError(f"Adapter factory {spec!r} did not return an object with .compile(...)")
        return _CallableAdapter(name=spec, fn=attr)
    raise TypeError(f"Adapter {spec!r} must be a class, adapter instance, or callable")


def adapter_for_model(model_name: str) -> CameraAdapter:
    key = canonical_model_key(model_name)
    if is_deferred_model(key):
        raise ValueError(f"Model {key} is deferred and excluded from unified camera registry")
    record = model_record(key)
    if key in _REGISTRY:
        cached_spec = _DYNAMIC_ADAPTER_SPECS.get(key)
        if cached_spec is None or cached_spec == record.adapter:
            return _REGISTRY[key]

    spec = str(record.adapter or "").strip()
    if spec in _NAMED_ADAPTERS:
        adapter = _NAMED_ADAPTERS[spec]
        _REGISTRY[key] = adapter
        _DYNAMIC_ADAPTER_SPECS[key] = spec
        return adapter
    if _looks_like_import_spec(spec):
        adapter = _resolve_import_adapter(spec)
        _REGISTRY[key] = adapter
        _DYNAMIC_ADAPTER_SPECS[key] = spec
        return adapter
    raise KeyError(f"No unified camera adapter registered for {key}")


def registered_model_keys() -> list[str]:
    active_by_key = {record.key: record for record in all_records() if not record.is_deferred}
    keys: set[str] = set()
    for key in list(_REGISTRY):
        if key in _DYNAMIC_ADAPTER_SPECS and key not in active_by_key:
            continue
        keys.add(key)
    for key, record in active_by_key.items():
        spec = str(record.adapter or "").strip()
        if spec in _NAMED_ADAPTERS or _looks_like_import_spec(spec):
            keys.add(key)
    return sorted(keys)


def compile_camera_payload(
    trajectory: CameraTrajectory,
    *,
    model_name: str,
    width: int,
    height: int,
    num_frames: int,
    work_dir: str | Path | None = None,
    device: str | None = None,
    prompt: str = "",
) -> CameraPayload:
    adapter = adapter_for_model(model_name)
    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "work_dir": work_dir,
        "device": device,
    }
    if "prompt" in inspect.signature(adapter.compile).parameters:
        kwargs["prompt"] = prompt
    return adapter.compile(trajectory, **kwargs)
