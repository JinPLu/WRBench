#!/usr/bin/env python3
"""WRBench quickstart example.

Demonstrates:
  (a) Listing supported models.
  (b) Compiling a yaw_LR preset for wan22-fun-5b-cam into a temp dir.
  (c) Compiling an arbitrary sweep('yaw','left',37,frames=49) script.
  (d) Printing the model-control timeline from the returned payload metadata.

Run from the repo root::

    PYTHONPATH=src python3 examples/quickstart.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import wrbench
from wrbench.datasets import natural25_first_frame_path


def _section(title: str) -> None:
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# (a) List models
# ---------------------------------------------------------------------------
_section("(a) Supported models")

models = wrbench.list_models(include_deferred=False)
print(f"Active models ({len(models)}):")
for key in models:
    rec = wrbench.model_record(key)
    print(
        f"  {key:<35} model_input={rec.model_input:<4} "
        f"runtime_input={rec.input_kind:<12} adapter={rec.adapter}"
    )

deferred = wrbench.list_models(include_deferred=True)
deferred_only = [k for k in deferred if k not in models]
if deferred_only:
    print(f"\nDeferred models ({len(deferred_only)}):")
    for key in deferred_only:
        print(f"  {key}")

# ---------------------------------------------------------------------------
# (b) Compile yaw_LR preset for wan22-fun-5b-cam
# ---------------------------------------------------------------------------
_section("(b) Compile preset yaw_LR → wan22-fun-5b-cam")

DEMO_MODEL = "wan22-fun-5b-cam"
DEMO_IMAGE = str(natural25_first_frame_path("bedroom_cat_bed_jump"))

with tempfile.TemporaryDirectory(prefix="wrbench_quickstart_") as tmpdir:
    out_path = str(Path(tmpdir) / "yaw_lr_demo.mp4")
    # Build the preset script programmatically
    preset_script = wrbench.presets.yaw_LR(peak_deg=60.0, frames=81)
    print(f"Preset script : {preset_script.to_string()}")
    print(f"Preset frames : {preset_script.frame_count}")

    result_b = wrbench.compile_camera(
        model=DEMO_MODEL,
        camera=preset_script,
        out=out_path,
        image=DEMO_IMAGE,
        dry_run=True,
    )

    artifacts: dict[str, str] = result_b["artifacts"]
    print(f"\nArtifact paths written to {tmpdir}:")
    for name, path in artifacts.items():
        exists = Path(path).exists()
        print(f"  {name}: {Path(path).name}  (exists={exists})")

    # Payload JSON sidecar
    payload_json_path = Path(out_path).with_suffix(".mp4.payload.json")
    if payload_json_path.exists():
        print(f"  payload_json: {payload_json_path.name}  (exists=True)")

    # ---------------------------------------------------------------------------
    # (c) Compile arbitrary sweep script
    # ---------------------------------------------------------------------------
    _section("(c) Compile sweep('yaw','left',37,frames=49)")

    sweep_script = wrbench.presets.sweep("yaw", "left", 37, frames=49)
    print(f"Sweep script  : {sweep_script.to_string()}")
    print(f"Sweep frames  : {sweep_script.frame_count}")

    out_path_c = str(Path(tmpdir) / "sweep_demo.mp4")
    result_c = wrbench.compile_camera(
        model=DEMO_MODEL,
        camera=sweep_script,
        out=out_path_c,
        image=DEMO_IMAGE,
        dry_run=True,
    )

    artifacts_c: dict[str, str] = result_c["artifacts"]
    print(f"\nArtifact paths:")
    for name, path in artifacts_c.items():
        exists = Path(path).exists()
        print(f"  {name}: {Path(path).name}  (exists={exists})")

    # ---------------------------------------------------------------------------
    # (d) Print model-control timeline from payload metadata
    # ---------------------------------------------------------------------------
    _section("(d) Model-control timeline")

    payload_c = result_c["payload"]
    timeline = payload_c.metadata.get("model_control_timeline", {})
    if timeline:
        print("model_control_timeline:")
        print(json.dumps(timeline, indent=2, default=str)[:2000])  # truncate long output
    else:
        print("model_control_timeline: (not present in metadata)")

    # Also show the model_control_samples sidecar if written
    samples_path = Path(out_path_c).with_suffix(".mp4.model_control_samples.json")
    if samples_path.exists():
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        print(f"\nmodel_control_samples sidecar keys: {list(samples.keys())}")
        inner_timeline = samples.get("model_control_timeline", {})
        if inner_timeline:
            preview = json.dumps(inner_timeline, indent=2, default=str)[:1000]
            print("  model_control_timeline (preview):")
            for line in preview.splitlines():
                print("   ", line)

print()
print("Quickstart complete.")
