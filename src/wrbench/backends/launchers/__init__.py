"""Registry of supported local subprocess launchers."""

from wrbench.backends.launchers._registry import LaunchSpec
from wrbench.backends.launchers.easyanimate import SPEC as EASYANIMATE, build_easyanimate_command
from wrbench.backends.launchers.minwm_hy import SPEC as MINWM_HY, build_minwm_hy_command
from wrbench.backends.launchers.minwm_wan import SPEC as MINWM_WAN, build_minwm_wan_command
from wrbench.backends.launchers.spatia import SPEC as SPATIA, build_spatia_command

_SPECS = {spec.key: spec for spec in (EASYANIMATE, MINWM_HY, MINWM_WAN, SPATIA)}


def launcher_spec(model: str) -> LaunchSpec | None:
    return _SPECS.get(model)


def supported_models() -> frozenset[str]:
    return frozenset(_SPECS)


__all__ = [
    "build_easyanimate_command",
    "build_minwm_hy_command",
    "build_minwm_wan_command",
    "build_spatia_command",
    "launcher_spec",
    "supported_models",
]
