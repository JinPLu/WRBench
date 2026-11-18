"""Backend selection for the evaluation stages."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


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


def _register(stage: str, name: str, backend: Backend) -> Backend:
    key = name.strip().lower()
    if not key or (not callable(backend) and not isinstance(backend, str)):
        raise ValueError("A backend requires a name and a callable or module:callable reference")
    _BACKENDS[stage][key] = backend
    return backend


def get_backend(stage: str, name: str | None = None) -> Callable[..., int]:
    """Resolve a registered backend or an explicit module:callable reference."""
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
    return sorted(_BACKENDS["pose"])


def register_segment_policy(name: str, policy: SegmentPolicyPlugin) -> SegmentPolicyPlugin:
    _SEGMENT_POLICIES[str(name).strip().lower()] = policy
    return policy


def registered_segment_policies() -> dict[str, SegmentPolicyPlugin]:
    return dict(_SEGMENT_POLICIES)


def register_d2_scorer(name: str, scorer: Backend) -> Backend:
    return _register("d2", name, scorer)


def registered_d2_scorers() -> list[str]:
    return sorted(_BACKENDS["d2"])


def register_vlm_scorer(name: str, scorer: Backend) -> Backend:
    return _register("vlm", name, scorer)


def registered_vlm_scorers() -> list[str]:
    return sorted(_BACKENDS["vlm"])


def register_builtin_plugins() -> None:
    """Populate default WRBench built-in plugins idempotently."""
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


register_builtin_plugins()
