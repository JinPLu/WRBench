"""Single source of truth for supported models.

Each model is declared by exactly one JSON file in ``wrbench/models/<key>.json``.
That file carries everything the toolkit needs: canonical key, aliases, input
kind (image vs source video vs no extra media), adapter name, amplitude calibration, capabilities,
and status. This replaces the historical multi-file config chain (separate alias
map, contracts, capabilities, amplitude, shell lists, run specs) with one
authoritative record per model, validated on load.

Adding a model = drop one JSON here + one adapter module. ``wrbench doctor``
checks that the two agree.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MODELS_DIR = Path(__file__).resolve().parent / "models"
PLUGIN_ENTRY_POINT_GROUP = "wrbench.plugins"

VALID_INPUT_KINDS = {"image", "none", "source_video"}
VALID_MODEL_INPUTS = {"T2V", "TI2V", "TV2V"}
VALID_VIEWPOINT_CONDITION_TYPES = {
    "geometry-cache",
    "model-inferred",
    "prompt-only",
    "source-video",
}
VALID_SOURCE_VIDEO_USAGES = {
    "none",
    "first_frame_extraction",
    "temporal_conditioning_source_video",
    "temporal_conditioning_via_gen3c_vipe_cache",
}
TEMPORAL_SOURCE_VIDEO_USAGES = {
    "temporal_conditioning_source_video",
    "temporal_conditioning_via_gen3c_vipe_cache",
}

VALID_TRANSLATION_UNITS = {
    "none",
    "canonical_scene",
    "cameractrl_scene",
    "blender_c2w_scene",
    "cache3d_scene",
    "relative_pose_embedding",
    "w2c_scene",
    "official_displacement_scale",
    "trajectory_template_c2w",
    "hydra_c2w_pre_div100",
}


class RegistryError(ValueError):
    pass


@dataclass(frozen=True)
class CameraAmplitude:
    model_key: str
    rotation_gain: float
    translation_gain: float
    max_amount: float
    translation_unit: str
    calibration_status: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ModelRecord:
    key: str
    aliases: tuple[str, ...]
    status: str
    input_kind: str
    model_input: str
    source_video_usage: str
    viewpoint_condition_type: str
    adapter: str
    payload_type: str
    amplitude: CameraAmplitude
    capabilities: dict[str, Any]
    notes: str
    default_width: int
    default_height: int
    default_frames: int
    default_fps: int
    execution_contract: dict[str, Any] | None = None

    @property
    def is_deferred(self) -> bool:
        return self.status != "active"


def _norm(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def _require_str(record: dict[str, Any], field: str, *, key: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{key}: missing required string field '{field}'")
    return value


def _require_int(record: dict[str, Any], field: str, *, key: str) -> int:
    value = record.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise RegistryError(f"{key}: missing required integer field '{field}'")
    return int(value)


def _require_float(record: dict[str, Any], field: str, *, key: str) -> float:
    value = record.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise RegistryError(f"{key}: missing required numeric field '{field}'")
    return float(value)


def register_translation_unit(unit: str) -> None:
    """Register an additional valid amplitude ``translation_unit``."""
    normalized = str(unit or "").strip()
    if not normalized:
        raise RegistryError("translation_unit must be a non-empty string")
    VALID_TRANSLATION_UNITS.add(normalized)


def _is_valid_translation_unit(unit: str) -> bool:
    if unit in VALID_TRANSLATION_UNITS:
        return True
    if unit.startswith("custom:"):
        return bool(unit.split(":", 1)[1].strip())
    return False


def _parse_amplitude(key: str, record: dict[str, Any]) -> CameraAmplitude:
    amp = record.get("amplitude")
    if amp is None and str(record.get("adapter") or "") == "prompt_camera":
        return CameraAmplitude(
            model_key=key,
            rotation_gain=1.0,
            translation_gain=1.0,
            max_amount=0.5,
            translation_unit="none",
            calibration_status="prompt_only",
            metadata={},
        )
    if not isinstance(amp, dict):
        raise RegistryError(f"{key}: missing 'amplitude' object")
    unit = _require_str(amp, "translation_unit", key=key)
    if not _is_valid_translation_unit(unit):
        raise RegistryError(f"{key}: unsupported translation_unit {unit!r}")
    try:
        return CameraAmplitude(
            model_key=key,
            rotation_gain=_require_float(amp, "rotation_gain", key=key),
            translation_gain=_require_float(amp, "translation_gain", key=key),
            max_amount=_require_float(amp, "max_amount", key=key),
            translation_unit=unit,
            calibration_status=_require_str(amp, "calibration_status", key=key),
            metadata=dict(amp.get("metadata", {}) or {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RegistryError(f"{key}: invalid amplitude record: {exc}") from exc


def _parse_model_input_metadata(
    *,
    key: str,
    status: str,
    input_kind: str,
    payload: dict[str, Any],
) -> tuple[str, str]:
    if status == "deferred":
        return str(payload.get("model_input") or ""), str(payload.get("source_video_usage") or "")

    defaults = {"none": "T2V", "image": "TI2V"}
    if input_kind == "source_video":
        if not payload.get("model_input"):
            raise RegistryError(
                f"{key}: source_video records must declare model_input explicitly "
                "(TV2V for temporal conditioning or TI2V for first-frame extraction)"
            )
        if not payload.get("source_video_usage"):
            raise RegistryError(f"{key}: source_video records must declare source_video_usage explicitly")

    model_input = str(payload.get("model_input") or defaults.get(input_kind, ""))
    source_video_usage = str(payload.get("source_video_usage") or "none")
    if model_input not in VALID_MODEL_INPUTS:
        raise RegistryError(f"{key}: model_input must be one of {sorted(VALID_MODEL_INPUTS)}")
    if source_video_usage not in VALID_SOURCE_VIDEO_USAGES:
        raise RegistryError(
            f"{key}: source_video_usage must be one of {sorted(VALID_SOURCE_VIDEO_USAGES)}"
        )

    if input_kind == "none" and (model_input, source_video_usage) != ("T2V", "none"):
        raise RegistryError(f"{key}: input_kind=none requires model_input='T2V' and source_video_usage='none'")
    if input_kind == "image" and (model_input, source_video_usage) != ("TI2V", "none"):
        raise RegistryError(f"{key}: input_kind=image requires model_input='TI2V' and source_video_usage='none'")
    if input_kind == "source_video":
        if model_input not in {"TI2V", "TV2V"}:
            raise RegistryError(f"{key}: input_kind=source_video requires model_input TI2V or TV2V")
        if source_video_usage == "first_frame_extraction" and model_input != "TI2V":
            raise RegistryError(f"{key}: first_frame_extraction requires model_input='TI2V'")
        if source_video_usage in TEMPORAL_SOURCE_VIDEO_USAGES and model_input != "TV2V":
            raise RegistryError(f"{key}: temporal source_video_usage requires model_input='TV2V'")
    return model_input, source_video_usage


def _parse_record(path: Path, payload: dict[str, Any]) -> ModelRecord:
    key = payload.get("key")
    if not isinstance(key, str) or not key:
        raise RegistryError(f"{path.name}: missing 'key'")
    status = _require_str(payload, "status", key=key)
    if status not in {"active", "deferred", "reference"}:
        raise RegistryError(f"{key}: status must be 'active', 'deferred', or 'reference'")
    input_kind = str(payload.get("input_kind") or "")
    if status == "active" and input_kind not in VALID_INPUT_KINDS:
        raise RegistryError(f"{key}: input_kind must be one of {sorted(VALID_INPUT_KINDS)}")
    adapter = str(payload.get("adapter") or "")
    if status == "active" and not adapter:
        raise RegistryError(f"{key}: active model requires 'adapter'")
    aliases = tuple(dict.fromkeys([_norm(key)] + [_norm(a) for a in payload.get("aliases", []) or []]))
    amplitude = _parse_amplitude(key, payload) if status == "active" else CameraAmplitude(
        model_key=key,
        rotation_gain=1.0,
        translation_gain=1.0,
        max_amount=0.5,
        translation_unit="canonical_scene",
        calibration_status="deferred",
        metadata={},
    )
    execution_contract = payload.get("execution_contract")
    if execution_contract is not None and not isinstance(execution_contract, dict):
        raise RegistryError(f"{key}: execution_contract must be an object")
    model_input, source_video_usage = _parse_model_input_metadata(
        key=key,
        status=status,
        input_kind=input_kind,
        payload=payload,
    )
    viewpoint_condition_type = str(payload.get("viewpoint_condition_type") or "")
    if viewpoint_condition_type and viewpoint_condition_type not in VALID_VIEWPOINT_CONDITION_TYPES:
        raise RegistryError(
            f"{key}: viewpoint_condition_type must be one of "
            f"{sorted(VALID_VIEWPOINT_CONDITION_TYPES)}"
        )
    if status == "reference" and not viewpoint_condition_type:
        raise RegistryError(f"{key}: reference records require viewpoint_condition_type")
    return ModelRecord(
        key=key,
        aliases=aliases,
        status=status,
        input_kind=input_kind,
        model_input=model_input,
        source_video_usage=source_video_usage,
        viewpoint_condition_type=viewpoint_condition_type,
        adapter=adapter,
        payload_type=_require_str(payload, "payload_type", key=key) if status == "active" else str(payload.get("payload_type") or ""),
        amplitude=amplitude,
        capabilities=dict(payload.get("capabilities", {}) or {}),
        notes=str(payload.get("notes") or ""),
        default_width=_require_int(payload, "default_width", key=key) if status == "active" else int(payload.get("default_width", 0)),
        default_height=_require_int(payload, "default_height", key=key) if status == "active" else int(payload.get("default_height", 0)),
        default_frames=_require_int(payload, "default_frames", key=key) if status == "active" else int(payload.get("default_frames", 0)),
        default_fps=_require_int(payload, "default_fps", key=key) if status == "active" else int(payload.get("default_fps", 0)),
        execution_contract=dict(execution_contract) if execution_contract else None,
    )


class _PluginAPI:
    """Lightweight plugin registration surface passed to 1-arg ``wrbench.plugins`` entry points."""

    def register_model_record(
        self,
        payload: dict[str, Any],
        *,
        source: str = "<plugin>",
        overwrite: bool = False,
    ) -> ModelRecord:
        return register_model_record(payload, source=source, overwrite=overwrite)

    def load_model_spec_file(
        self,
        path: str | Path,
        *,
        overwrite: bool = True,
    ) -> ModelRecord:
        return load_model_spec_file(path, overwrite=overwrite)

    def register_translation_unit(self, unit: str) -> None:
        register_translation_unit(unit)

    def register_adapter(self, model_name_or_group: str | list[str] | tuple[str, ...], adapter: Any) -> None:
        from wrbench.adapters.base import register_adapter

        register_adapter(model_name_or_group, adapter)

    def register_generation_backend(self, name: str, factory: Any) -> Any:
        from wrbench.backends.registry import register_generation_backend

        return register_generation_backend(name, factory)

    def register_launcher(self, spec: Any) -> Any:
        from wrbench.backends.launchers import register_launcher

        return register_launcher(spec)


_CACHE: tuple[dict[str, ModelRecord], dict[str, str]] | None = None
_LOADING = False


def _insert_record(
    records: dict[str, ModelRecord],
    alias_lookup: dict[str, str],
    record: ModelRecord,
    *,
    overwrite: bool,
) -> ModelRecord:
    existing = records.get(record.key)
    if existing is not None and not overwrite:
        raise RegistryError(f"Duplicate model key: {record.key}")
    for alias in record.aliases:
        owner = alias_lookup.get(alias)
        if owner is not None and owner != record.key:
            raise RegistryError(f"Alias collision {alias!r}: {owner} vs {record.key}")
    if existing is not None:
        for old_alias in existing.aliases:
            if alias_lookup.get(old_alias) == record.key:
                del alias_lookup[old_alias]
    records[record.key] = record
    for alias in record.aliases:
        alias_lookup[alias] = record.key
    return record


def _iter_model_dirs() -> list[Path]:
    raw = os.environ.get("WRBENCH_MODEL_DIRS", "")
    dirs: list[Path] = []
    for entry in raw.split(os.pathsep):
        cleaned = entry.strip()
        if not cleaned:
            continue
        candidate = Path(cleaned).expanduser()
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs


def _entry_points_for_group(group: str) -> list[Any]:
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group=group))
    return list(eps.get(group, ()))


def _invoke_plugin_entry_point(ep: Any) -> None:
    fn = ep.load()
    if not callable(fn):
        return
    sig = inspect.signature(fn)
    required_positional = [
        p
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not required_positional:
        fn()
    elif len(required_positional) == 1:
        fn(_PluginAPI())
    else:
        raise RegistryError(
            f"Plugin entry point {getattr(ep, 'name', ep)!r} expects {len(required_positional)} "
            "required arguments; expected 0 or 1"
        )


def _load() -> tuple[dict[str, ModelRecord], dict[str, str]]:
    global _CACHE, _LOADING
    if _CACHE is not None:
        return _CACHE
    if not MODELS_DIR.exists():
        raise RegistryError(f"Missing models directory: {MODELS_DIR}")
    records: dict[str, ModelRecord] = {}
    alias_lookup: dict[str, str] = {}
    _CACHE = (records, alias_lookup)
    _LOADING = True
    try:
        for path in sorted(MODELS_DIR.glob("*.json")):
            record = _parse_record(path, json.loads(path.read_text(encoding="utf-8")))
            _insert_record(records, alias_lookup, record, overwrite=False)
        if not records:
            raise RegistryError(f"No model records found in {MODELS_DIR}")
        for extra_dir in _iter_model_dirs():
            for path in sorted(extra_dir.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    raise RegistryError(f"{path}: model spec must be a JSON object")
                record = _parse_record(path, payload)
                _insert_record(records, alias_lookup, record, overwrite=True)
        for ep in _entry_points_for_group(PLUGIN_ENTRY_POINT_GROUP):
            _invoke_plugin_entry_point(ep)
    except Exception:
        _CACHE = None
        raise
    finally:
        _LOADING = False
    return records, alias_lookup


def register_model_record(
    payload: dict[str, Any],
    *,
    source: str = "<runtime>",
    overwrite: bool = False,
) -> ModelRecord:
    """Validate and register a model record dynamically without modifying wrbench source."""
    if not isinstance(payload, dict):
        raise RegistryError(f"{source}: model record payload must be a dict")
    records, alias_lookup = _load()
    record = _parse_record(Path(source), payload)
    return _insert_record(records, alias_lookup, record, overwrite=overwrite)


def load_model_spec_file(
    path: str | Path,
    *,
    overwrite: bool = True,
) -> ModelRecord:
    """Load and register a model record from an external JSON spec file."""
    resolved = Path(path).expanduser()
    if not resolved.is_file():
        raise FileNotFoundError(f"Model spec file not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RegistryError(f"{resolved}: model spec must be a JSON object")
    return register_model_record(payload, source=str(resolved), overwrite=overwrite)


def reload() -> None:
    """Clear the cached registry and re-read built-in models, WRBENCH_MODEL_DIRS, and plugins on next load."""
    global _CACHE, _LOADING
    _CACHE = None
    _LOADING = False


def canonical_model_key(name: str) -> str:
    _, lookup = _load()
    key = _norm(name)
    if key in lookup:
        return lookup[key]
    raise KeyError(f"Unknown model key or alias: {name}")


def model_record(name: str) -> ModelRecord:
    records, _ = _load()
    return records[canonical_model_key(name)]


def all_records() -> list[ModelRecord]:
    records, _ = _load()
    return [records[k] for k in sorted(records)]


def active_model_keys() -> list[str]:
    return [r.key for r in all_records() if not r.is_deferred]


def deferred_model_keys() -> list[str]:
    return [r.key for r in all_records() if r.is_deferred]


def is_deferred_model(name: str) -> bool:
    return model_record(name).is_deferred


def input_kind(name: str) -> str:
    return model_record(name).input_kind


def model_input(name: str) -> str:
    return model_record(name).model_input


def source_video_usage(name: str) -> str:
    return model_record(name).source_video_usage


def viewpoint_condition_type(name: str) -> str:
    value = model_record(name).viewpoint_condition_type
    if not value:
        raise RegistryError(f"{name}: viewpoint_condition_type is not declared")
    return value


def adapter_name(name: str) -> str:
    return model_record(name).adapter


def resolve_model_amplitude(name: str) -> CameraAmplitude:
    return model_record(name).amplitude
