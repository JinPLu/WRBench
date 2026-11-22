"""Command-line interface entrypoint for wrbench.

Sub-commands
------------
models           List supported models.
presets          List preset camera motions.
actions          Parse and validate a frame-action camera script.
plugins          List models, adapters, generation backends, launchers, and eval plugins.
generate         Compile a camera script and write model-native payload sidecars.
doctor           Validate registry, adapters, backends, and eval runtime.
tasks            Export standardized Natural-25 task packages for external generation.
eval             WRBench diagnostic evaluation (D1-D6, validation, contract, plugins).
profile          Profile a generation command (wall time, GPU memory, stage spans).
profile-summary  Aggregate resource profile JSON files.
prompt           Generate scene/task/camera prompts.
firstframe       Generate first-frame images from T2I prompts.
"""

from __future__ import annotations

import argparse
import sys

from wrbench import __version__
from wrbench._cli.common import (
    _col,
    _print_table,
    add_model_spec_argument,
    load_cli_model_specs,
)
from wrbench._cli.content_cmd import (
    _cmd_firstframe,
    _cmd_profile,
    _cmd_profile_summary,
    _cmd_prompt,
    register_content_parsers,
)
from wrbench._cli.doctor_cmd import (
    _cmd_doctor,
    _doctor_check_model,
    register_doctor_parser,
)
from wrbench._cli.eval_cmd import _cmd_eval, register_eval_parser
from wrbench._cli.generate_cmd import (
    _cmd_generate,
    _resolve_preset_camera,
    register_generate_parser,
)
from wrbench._cli.models_cmd import (
    _cmd_actions,
    _cmd_models,
    _cmd_plugins,
    _cmd_presets,
    register_models_parsers,
)
from wrbench._cli.tasks_cmd import _cmd_tasks, register_tasks_parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wrbench",
        description="Unified camera control and diagnostic evaluation for video world models.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    add_model_spec_argument(parser, dest="model_specs")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    register_models_parsers(sub)
    register_generate_parser(sub)
    register_content_parsers(sub)
    register_doctor_parser(sub)
    register_tasks_parser(sub)
    register_eval_parser(sub)

    return parser


build_parser = _build_parser


_DISPATCH = {
    "models": _cmd_models,
    "presets": _cmd_presets,
    "actions": _cmd_actions,
    "plugins": _cmd_plugins,
    "generate": _cmd_generate,
    "doctor": _cmd_doctor,
    "tasks": _cmd_tasks,
    "profile": _cmd_profile,
    "profile-summary": _cmd_profile_summary,
    "prompt": _cmd_prompt,
    "firstframe": _cmd_firstframe,
    "eval": _cmd_eval,
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        load_cli_model_specs(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: failed to load --model-spec: {exc}", file=sys.stderr)
        return 1

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    if handler in (_cmd_eval, _cmd_tasks):
        try:
            return handler(args)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    return handler(args)


__all__ = [
    "main",
    "build_parser",
    "_build_parser",
    "_doctor_check_model",
    "_col",
    "_print_table",
    "_resolve_preset_camera",
    "_cmd_models",
    "_cmd_presets",
    "_cmd_actions",
    "_cmd_plugins",
    "_cmd_generate",
    "_cmd_doctor",
    "_cmd_tasks",
    "_cmd_profile",
    "_cmd_profile_summary",
    "_cmd_prompt",
    "_cmd_firstframe",
    "_cmd_eval",
    "_DISPATCH",
]


if __name__ == "__main__":
    raise SystemExit(main())
