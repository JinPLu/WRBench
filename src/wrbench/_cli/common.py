"""Shared formatting, model-spec loading, and plugin summary helpers for the CLI."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _col(value: Any, width: int) -> str:
    """Left-justify *value* in a field of *width* characters."""
    return str(value).ljust(width)


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    """Print a simple fixed-width table to stdout."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    sep = "  ".join("-" * w for w in widths)
    header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print(sep)
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def add_model_spec_argument(
    parser: argparse.ArgumentParser,
    *,
    dest: str = "sub_model_specs",
) -> None:
    """Register repeatable ``--model-spec`` on a parser without clobbering parent values."""
    parser.add_argument(
        "--model-spec",
        action="append",
        default=[],
        dest=dest,
        metavar="PATH",
        help="Load an external model spec JSON file (repeatable).",
    )


def collect_model_specs(args: argparse.Namespace) -> list[str]:
    """Return all ``--model-spec`` paths supplied on root or subcommand parsers."""
    root_specs = list(getattr(args, "model_specs", None) or [])
    sub_specs = list(getattr(args, "sub_model_specs", None) or [])
    return root_specs + sub_specs


def load_cli_model_specs(args: argparse.Namespace) -> list[str]:
    """Load any ``--model-spec`` JSON files supplied on the command line."""
    spec_paths = collect_model_specs(args)
    if not spec_paths:
        return []

    from wrbench.registry import load_model_spec_file

    loaded_keys: list[str] = []
    for raw_path in spec_paths:
        record = load_model_spec_file(raw_path, overwrite=True)
        loaded_keys.append(record.key)
    return loaded_keys


def collect_plugins_summary() -> dict[str, Any]:
    """Collect active/custom models, adapters, backends, launchers, and eval plugins."""
    import wrbench
    import wrbench.adapters  # noqa: F401
    from wrbench.adapters.base import registered_model_keys
    from wrbench.backends import registry as backend_registry
    from wrbench.backends.launchers import supported_models as registered_launchers
    from wrbench.eval.plugins import (
        registered_d2_scorers,
        registered_pose_backends,
        registered_segment_policies,
        registered_vlm_scorers,
    )

    active_models = wrbench.active_model_keys()
    deferred_models = wrbench.deferred_model_keys()

    # Gather declarative & registered named adapter names
    adapter_names: set[str] = set()
    from wrbench.adapters import base as adapters_base

    if hasattr(adapters_base, "registered_named_adapters"):
        adapter_names.update(adapters_base.registered_named_adapters())
    for rec in wrbench.all_records():
        if rec.adapter:
            adapter_names.add(rec.adapter)

    if hasattr(backend_registry, "registered_generation_backends"):
        generation_backends = sorted(backend_registry.registered_generation_backends())
    else:
        generation_backends = ["dry_run", "local_subprocess"]

    return {
        "models": {
            "active": active_models,
            "deferred": deferred_models,
        },
        "generation": {
            "adapters": sorted(adapter_names),
            "adapter_models": registered_model_keys(),
            "backends": generation_backends,
            "launchers": sorted(registered_launchers()),
        },
        "eval": {
            "pose_backends": registered_pose_backends(),
            "segment_policies": sorted(registered_segment_policies()),
            "d2_scorers": registered_d2_scorers(),
            "vlm_scorers": registered_vlm_scorers(),
        },
    }


def print_plugins_summary(payload: dict[str, Any], *, as_json: bool) -> int:
    """Print the plugins summary in JSON or human-readable format."""
    if as_json:
        print(json.dumps(payload, indent=2))
        return 0

    for section, mapping in payload.items():
        print(f"[{section}]")
        for key, items in mapping.items():
            print(f"  {key}: {', '.join(items) if items else 'none'}")
    return 0


def print_cli_error(message: str) -> int:
    """Write a standardized error message to stderr and return exit code 1."""
    print(f"error: {message}", file=sys.stderr)
    return 1
