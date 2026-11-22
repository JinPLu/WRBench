"""Registry of supported local subprocess launchers."""

from wrbench.backends.launchers._registry import LaunchSpec, PreparedLaunch
from wrbench.backends.launchers.easyanimate import SPEC as EASYANIMATE, build_easyanimate_command
from wrbench.backends.launchers.minwm_hy import SPEC as MINWM_HY, build_minwm_hy_command
from wrbench.backends.launchers.minwm_wan import SPEC as MINWM_WAN, build_minwm_wan_command
from wrbench.backends.launchers.recammaster import SPEC as RECAMMASTER, build_recammaster_command
from wrbench.backends.launchers.spatia import SPEC as SPATIA, build_spatia_command

_SPECS = {spec.key: spec for spec in (EASYANIMATE, MINWM_HY, MINWM_WAN, RECAMMASTER, SPATIA)}


def register_launcher(spec: LaunchSpec) -> LaunchSpec:
    """Register a custom ``LaunchSpec`` for ``LocalSubprocessBackend``."""
    if not isinstance(spec, LaunchSpec):
        raise TypeError(f"Expected LaunchSpec, got {type(spec).__name__}")
    key = str(spec.key or "").strip()
    if not key:
        raise ValueError("LaunchSpec.key must be a non-empty string")
    _SPECS[key] = spec
    return spec


def launcher_spec(model: str) -> LaunchSpec | None:
    return _SPECS.get(model)


def supported_models() -> frozenset[str]:
    return frozenset(_SPECS)


__all__ = [
    "LaunchSpec",
    "PreparedLaunch",
    "build_easyanimate_command",
    "build_minwm_hy_command",
    "build_minwm_wan_command",
    "build_recammaster_command",
    "build_spatia_command",
    "launcher_spec",
    "register_launcher",
    "supported_models",
]
