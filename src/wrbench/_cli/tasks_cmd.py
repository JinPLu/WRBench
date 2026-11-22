"""CLI subcommand: tasks (standardized task package export)."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from wrbench._cli.common import add_model_spec_argument


def _parse_camera_presets(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


def _cmd_tasks(args: argparse.Namespace) -> int:
    if args.tasks_command == "export":
        from wrbench.tasks_submission import export_benchmark_tasks

        camera_presets = _parse_camera_presets(args.cameras)
        result = export_benchmark_tasks(
            out_dir=args.out_dir,
            model=args.model,
            release_id=args.release,
            camera_presets=camera_presets,
            limit=args.limit,
            compile_targets=not args.no_compile_targets,
        )
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"tasks_count : {result['tasks_count']}")
            print(f"tasks_jsonl : {result['tasks_jsonl']}")
            print(f"out_dir     : {result['out_dir']}")
        return 0

    print(f"tasks: unknown sub-command {args.tasks_command!r}", file=sys.stderr)
    return 2


def register_tasks_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the tasks subparser."""
    p_tasks = sub.add_parser(
        "tasks",
        help="Export standardized benchmark task packages for external video generation.",
    )
    add_model_spec_argument(p_tasks)
    tasks_sub = p_tasks.add_subparsers(dest="tasks_command", metavar="<tasks-command>")
    tasks_sub.required = True

    p_export = tasks_sub.add_parser(
        "export",
        help="Export Natural-25 benchmark tasks (prompts, first frames, and optional compiled camera targets).",
    )
    p_export.add_argument(
        "--out-dir",
        required=True,
        dest="out_dir",
        metavar="DIR",
        help="Output directory for tasks.jsonl and optional compiled targets.",
    )
    p_export.add_argument(
        "--model",
        metavar="MODEL",
        help="Optional model key to compile model-native camera targets alongside tasks.jsonl.",
    )
    p_export.add_argument(
        "--release",
        metavar="ID",
        help="Optional Natural-25 release ID (defaults to bundled release/toolkit catalog).",
    )
    p_export.add_argument(
        "--cameras",
        metavar="LIST",
        help="Comma-separated camera presets (e.g. static,yaw_LR,yaw_RL).",
    )
    p_export.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="Limit the number of exported tasks.",
    )
    p_export.add_argument(
        "--no-compile-targets",
        action="store_true",
        dest="no_compile_targets",
        help="Skip compiling camera target/payload sidecars even when --model is specified.",
    )
    p_export.add_argument(
        "--json",
        action="store_true",
        help="Print export summary as JSON.",
    )
    add_model_spec_argument(p_export)
