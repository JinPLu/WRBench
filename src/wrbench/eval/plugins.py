"""Backend selection for the evaluation stages."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib import metadata as importlib_metadata
import inspect
import sys
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
import warnings


DEFAULT_D1_SCORER_CONFIG: dict[str, Any] = {
    "pose_backend": "vggt_omega",
    "poses_file": "poses.npy",
    "default_frames": 121,
    "predicted_pose_type": "c2w",
    "predicted_camera_convention": "opencv",
    "target_camera_convention": "opencv",
    "rot_scale_deg": 45.0,
    "trans_scale": 1.0,
    "yaw_weak_threshold_deg": 2.0,
    "pan_weak_threshold": 0.0001,
    "static_rot_threshold_deg": 2.0,
    "static_trans_threshold": 0.05,
    "target_fallback_mode": "none",
}
D1_TARGET_MODES = ("none", "nominal_profile", "requested_profile")
D1_TARGET_MODE_HELP = (
    "D1 target mode. 'none' requires an explicit target; 'nominal_profile' adds a diagnostic "
    "when no target is supplied; 'requested_profile' scores against the requested camera profile."
)


@runtime_checkable
class SegmentPolicyPlugin(Protocol):
    name: str

    def applies(self, row: Mapping[str, Any]) -> bool:
        ...

    def slice_poses(self, row: Mapping[str, Any], poses: Any) -> tuple[Any, dict[str, Any]]:
        ...


@dataclass(frozen=True)
class ConcatSegmentPolicy:
    """Parameterized segment policy for models emitting source + generated concatenation."""

    name: str
    models: frozenset[str]
    condition_frames: int
    generated_frames: int

    def applies(self, row: Mapping[str, Any]) -> bool:
        return str(row.get("model") or "").strip().lower() in self.models

    def slice_poses(self, row: Mapping[str, Any], poses: Any) -> tuple[Any, dict[str, Any]]:
        from wrbench.eval.d1.hydra_segments import slice_concat_poses

        return slice_concat_poses(row, poses, policy=self)


_SEGMENT_POLICIES: dict[str, SegmentPolicyPlugin] = {}
Backend = Callable[..., int] | str
DEFAULT_BACKENDS = {"pose": "vggt_omega", "d2": "dinov2_fullfov_v2", "vlm": "qwen_runtime_v2"}
_BACKENDS: dict[str, dict[str, Backend]] = {stage: {} for stage in DEFAULT_BACKENDS}
EVAL_PLUGIN_ENTRY_POINT_GROUP = "wrbench.eval_plugins"
_ENTRY_POINTS_LOADED = False


def _iter_eval_entry_points() -> list[Any]:
    try:
        eps = importlib_metadata.entry_points(group=EVAL_PLUGIN_ENTRY_POINT_GROUP)
    except TypeError:
        all_eps = importlib_metadata.entry_points()
        if hasattr(all_eps, "select"):
            eps = all_eps.select(group=EVAL_PLUGIN_ENTRY_POINT_GROUP)
        elif isinstance(all_eps, Mapping):
            eps = all_eps.get(EVAL_PLUGIN_ENTRY_POINT_GROUP, ())
        else:
            eps = ()
    return list(eps)


def load_eval_entry_points(*, force: bool = False) -> list[str]:
    """Discover and invoke ``wrbench.eval_plugins`` entry points once."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED and not force:
        return []
    _ENTRY_POINTS_LOADED = True
    loaded: list[str] = []
    for ep in _iter_eval_entry_points():
        try:
            target = ep.load()
            if callable(target):
                sig = inspect.signature(target)
                required_positional = [
                    p
                    for p in sig.parameters.values()
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                    and p.default is inspect.Parameter.empty
                ]
                if not required_positional:
                    target()
                elif len(required_positional) == 1:
                    target(sys.modules[__name__])
                else:
                    raise TypeError(
                        f"entry point {ep.name!r} requires {len(required_positional)} positional arguments"
                    )
            loaded.append(str(getattr(ep, "name", ep)))
        except Exception as exc:
            warnings.warn(
                f"Failed to load {EVAL_PLUGIN_ENTRY_POINT_GROUP} entry point "
                f"{getattr(ep, 'name', ep)!r}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    return loaded


def _register(stage: str, name: str, backend: Backend) -> Backend:
    key = name.strip().lower()
    if not key or (not callable(backend) and not isinstance(backend, str)):
        raise ValueError("A backend requires a name and a callable or module:callable reference")
    _BACKENDS[stage][key] = backend
    return backend


def get_backend(stage: str, name: str | None = None) -> Callable[..., int]:
    """Resolve a registered backend or an explicit module:callable reference."""
    load_eval_entry_points()
    selected = name or DEFAULT_BACKENDS[stage]
    backend = _BACKENDS[stage].get(selected.lower(), selected)
    if callable(backend):
        return backend
    module, sep, attribute = backend.partition(":")
    if not sep or not module or not attribute:
        choices = ", ".join(sorted(_BACKENDS[stage]))
        raise ValueError(f"Unknown {stage} backend {selected!r}. Use {choices} or module:callable.")
    try:
        target = getattr(import_module(module), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot load {stage} backend {backend!r}: {exc}") from exc
    if not callable(target):
        raise ValueError(f"{stage} backend {backend!r} is not callable")
    return target


def register_pose_backend(name: str, backend: Backend) -> Backend:
    return _register("pose", name, backend)


def get_pose_backend(name: str) -> Callable[..., int]:
    return get_backend("pose", name)


def registered_pose_backends() -> list[str]:
    load_eval_entry_points()
    return sorted(_BACKENDS["pose"])


def register_segment_policy(name: str, policy: SegmentPolicyPlugin) -> SegmentPolicyPlugin:
    _SEGMENT_POLICIES[str(name).strip().lower()] = policy
    return policy


def registered_segment_policies() -> dict[str, SegmentPolicyPlugin]:
    load_eval_entry_points()
    return dict(_SEGMENT_POLICIES)


def register_d2_scorer(name: str, scorer: Backend) -> Backend:
    return _register("d2", name, scorer)


def registered_d2_scorers() -> list[str]:
    load_eval_entry_points()
    return sorted(_BACKENDS["d2"])


def register_vlm_scorer(name: str, scorer: Backend) -> Backend:
    return _register("vlm", name, scorer)


def registered_vlm_scorers() -> list[str]:
    load_eval_entry_points()
    return sorted(_BACKENDS["vlm"])


def register_builtin_plugins() -> None:
    """Populate default WRBench built-in plugins and discover entry points idempotently."""
    register_segment_policy(
        "hydra_concat_77_77",
        ConcatSegmentPolicy(
            name="hydra_concat_77_77",
            models=frozenset({"hydra"}),
            condition_frames=77,
            generated_frames=77,
        ),
    )
    register_pose_backend("vggt_omega", "wrbench.eval.runtime:_run_vggt_omega")
    register_d2_scorer("dinov2_fullfov_v2", "wrbench.eval.runtime:_run_dinov2_fullfov")
    register_vlm_scorer("qwen_runtime_v2", "wrbench.eval.runtime:_run_qwen_runtime")
    load_eval_entry_points(force=True)


register_builtin_plugins()
