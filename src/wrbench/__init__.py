"""WRBench: unified camera control, pluggable generation, and D1-D6 evaluation for video world models.

Quickstart::

    import wrbench
    from wrbench.datasets import natural25_first_frame_path

    # Preset combination (compile-only by default)
    wrbench.compile_camera(
        model="wan22-fun-5b-cam",
        camera="yaw:left:60@40,yaw:right:60@41",
        image=natural25_first_frame_path("bedroom_cat_bed_jump"),
        out="out.mp4",
    )

    # Arbitrary angle sweep
    script = wrbench.presets.sweep("yaw", "left", 37, frames=49)
    wrbench.compile_camera(
        model="wan22-fun-5b-cam",
        camera=script,
        image=natural25_first_frame_path("bedroom_cat_bed_jump"),
        out="out.mp4",
    )

With ``dry_run=True`` this compiles the model-native payload and writes auditable
sidecars without loading a model. See ``wrbench.backends`` for real generation
hooks and ``wrbench.eval`` for D1-D6 diagnostic scoring.
"""

from __future__ import annotations

from wrbench import presets
from wrbench.actions import CameraScript, FrameAction, parse_camera_script
from wrbench.builder import build_camera_trajectory
from wrbench.registry import (
    active_model_keys,
    all_records,
    canonical_model_key,
    deferred_model_keys,
    load_model_spec_file,
    model_record,
    register_model_record,
)
from wrbench.runner import compile_camera

__version__ = "0.1.3"


def list_models(include_deferred: bool = False) -> list[str]:
    """Canonical keys of supported models."""
    keys = active_model_keys()
    if include_deferred:
        keys = keys + deferred_model_keys()
    return keys


__all__ = [
    "__version__",
    "CameraScript",
    "FrameAction",
    "parse_camera_script",
    "build_camera_trajectory",
    "compile_camera",
    "presets",
    "list_models",
    "canonical_model_key",
    "model_record",
    "all_records",
    "active_model_keys",
    "deferred_model_keys",
    "load_model_spec_file",
    "register_model_record",
]
