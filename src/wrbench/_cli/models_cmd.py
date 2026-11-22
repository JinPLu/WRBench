"""CLI subcommands: models, presets, actions, and plugins."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from wrbench._cli.common import (
    _print_table,
    add_model_spec_argument,
    collect_plugins_summary,
    print_plugins_summary,
)


def _cmd_models(args: argparse.Namespace) -> int:
    import wrbench

    records = wrbench.all_records()
    if not args.deferred:
        records = [r for r in records if not r.is_deferred]

    if args.json:
        data = []
        for r in records:
            data.append(
                {
                    "key": r.key,
                    "aliases": list(r.aliases),
                    "status": r.status,
                    "input_kind": r.input_kind,
                    "model_input": r.model_input,
                    "source_video_usage": r.source_video_usage,
                    "viewpoint_condition_type": r.viewpoint_condition_type,
                    "adapter": r.adapter,
                    "payload_type": r.payload_type,
                    "calibration_status": r.amplitude.calibration_status,
                    "rotation_gain": r.amplitude.rotation_gain,
                    "translation_gain": r.amplitude.translation_gain,
                    "max_amount": r.amplitude.max_amount,
                    "translation_unit": r.amplitude.translation_unit,
                    "capabilities": r.capabilities,
                    "notes": r.notes,
                    "is_deferred": r.is_deferred,
                }
            )
        print(json.dumps(data, indent=2))
        return 0

    rows = [
        [
            r.key,
            r.model_input or "-",
            r.source_video_usage or "-",
            r.viewpoint_condition_type or "-",
            r.adapter or "-",
        ]
        for r in records
    ]
    _print_table(
        rows,
        ["key", "model_input", "source_usage", "viewpoint_condition", "adapter"],
    )
    return 0


def _cmd_presets(args: argparse.Namespace) -> int:
    import wrbench

    names = wrbench.presets.preset_names()
    descriptions = {
        "static": "Hold the camera still for the entire clip.",
        "yaw_LR": "Yaw left to peak_deg, then return right (go-return).",
        "yaw_RL": "Yaw right to peak_deg, then return left (go-return).",
        "pan_LR": "Pan left then return right (go-return).",
        "pan_RL": "Pan right then return left (go-return).",
    }

    if args.json:
        data = []
        for name in names:
            script = wrbench.presets.build_preset(name)
            data.append(
                {
                    "name": name,
                    "description": descriptions.get(name, ""),
                    "default_script": script.to_string(),
                    "default_frame_count": script.frame_count,
                }
            )
        print(json.dumps(data, indent=2))
        return 0

    rows = []
    for name in names:
        script = wrbench.presets.build_preset(name)
        desc = descriptions.get(name, "")
        rows.append([name, desc, script.to_string()])
    _print_table(rows, ["name", "description", "default_expansion"])
    return 0


def _cmd_actions(args: argparse.Namespace) -> int:
    import wrbench

    camera_text: str = args.camera
    fps: int = args.fps

    try:
        script = wrbench.parse_camera_script(camera_text, fps=fps)
    except (ValueError, KeyError) as exc:
        print(f"error: invalid camera script: {exc}", file=sys.stderr)
        return 1

    normalized = script.to_string()
    frame_count = script.frame_count

    segments = []
    for action in script.actions:
        seg: dict[str, Any] = {
            "kind": action.kind,
            "direction": action.direction,
            "frames": action.frames,
        }
        if action.degrees is not None:
            seg["value"] = action.degrees
            seg["value_unit"] = "degrees"
        elif action.amount is not None:
            seg["value"] = action.amount
            seg["value_unit"] = "amount"
        else:
            seg["value"] = None
            seg["value_unit"] = None
        segments.append(seg)

    if args.json:
        print(
            json.dumps(
                {
                    "normalized_script": normalized,
                    "frame_count": frame_count,
                    "fps": fps,
                    "segments": segments,
                },
                indent=2,
            )
        )
        return 0

    print(f"normalized : {normalized}")
    print(f"frame_count: {frame_count if frame_count is not None else '(unset)'}")
    print(f"fps        : {fps}")
    print()
    rows = []
    for i, seg in enumerate(segments):
        value_str = str(seg["value"]) if seg["value"] is not None else "-"
        unit_str = seg["value_unit"] or "-"
        rows.append([str(i), seg["kind"], seg["direction"], value_str, unit_str, str(seg["frames"])])
    _print_table(rows, ["#", "kind", "direction", "value", "unit", "frames"])
    return 0


def _cmd_plugins(args: argparse.Namespace) -> int:
    payload = collect_plugins_summary()
    return print_plugins_summary(payload, as_json=bool(args.json))


def register_models_parsers(sub: argparse._SubParsersAction[Any]) -> None:
    """Register models, presets, actions, and plugins subparsers."""
    p_models = sub.add_parser("models", help="List supported models.")
    p_models.add_argument(
        "--json", action="store_true", help="Output full records as a JSON array."
    )
    p_models.add_argument(
        "--deferred",
        action="store_true",
        help="Include deferred and reference-only (not dispatchable) models.",
    )
    add_model_spec_argument(p_models)

    p_presets = sub.add_parser("presets", help="List preset camera motions.")
    p_presets.add_argument("--json", action="store_true", help="Output as JSON.")

    p_actions = sub.add_parser(
        "actions", help="Parse and validate a frame-action camera script."
    )
    p_actions.add_argument(
        "--camera",
        required=True,
        metavar="SCRIPT",
        help='Camera script, e.g. "yaw:left:60@40,yaw:right:60@41".',
    )
    p_actions.add_argument(
        "--fps", type=int, required=True, metavar="N", help="Frames per second."
    )
    p_actions.add_argument("--json", action="store_true", help="Output as JSON.")

    p_plugins = sub.add_parser(
        "plugins", help="List active models, adapters, generation backends, launchers, and eval plugins."
    )
    p_plugins.add_argument("--json", action="store_true", help="Output as JSON.")
    add_model_spec_argument(p_plugins)
