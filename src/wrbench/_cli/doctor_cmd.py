"""CLI subcommand: doctor (model registry, adapter, backend, and eval runtime checks)."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from wrbench._cli.common import add_model_spec_argument


def _runtime_has_eval_block(runtime_config: Path | None) -> bool:
    """Return True if *runtime_config* exists and contains a non-empty ``eval`` block."""
    if runtime_config is None or not runtime_config.is_file():
        return False
    try:
        payload = json.loads(runtime_config.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(payload, dict) and bool(payload.get("eval"))


def _doctor_check_model(
    key: str,
    *,
    runtime: Any = None,
) -> tuple[bool, list[str]]:
    """Run all doctor checks for one model key. Returns (pass, [lines])."""
    lines: list[str] = []

    import wrbench
    import wrbench.adapters  # noqa: F401
    from wrbench.adapters.base import adapter_for_model

    # Check 1: registry record loads
    try:
        record = wrbench.model_record(key)
    except Exception as exc:
        lines.append(f"  FAIL  registry: {exc}")
        return False, lines

    lines.append(f"  pass  registry: record loaded (status={record.status})")

    # Deferred models: that is a pass for doctor
    if record.is_deferred:
        lines.append("  pass  deferred: model is deferred/not-dispatchable (expected)")
        lines.append(f"  PASS  {key}  [deferred]")
        return True, lines

    # Check 2: input_kind is valid
    from wrbench.registry import VALID_INPUT_KINDS

    if record.input_kind not in VALID_INPUT_KINDS:
        lines.append(f"  FAIL  input_kind: {record.input_kind!r} is not valid")
        return False, lines
    lines.append(f"  pass  input_kind: {record.input_kind}")

    # Check 3: adapter resolves (supports built-in, declarative, and module:callable adapters)
    try:
        adapter = adapter_for_model(key)
        lines.append(f"  pass  adapter: {type(adapter).__name__}")
    except Exception as exc:
        lines.append(f"  FAIL  adapter: {exc}")
        return False, lines

    # Check 4: dry-run compile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "doctor_test.mp4")
        dummy_image = os.path.join(tmp, "dummy_image.png")
        dummy_video = os.path.join(tmp, "dummy_video.mp4")
        Path(dummy_image).write_bytes(b"")
        Path(dummy_video).write_bytes(b"")

        caps = record.capabilities or {}
        if caps.get("supports_static", True):
            camera = f"static@{int(record.default_frames)}"
        else:
            camera = wrbench.presets.yaw_LR(
                peak_deg=30, frames=int(record.default_frames)
            ).to_string()

        compile_kwargs: dict[str, Any] = {
            "model": key,
            "camera": camera,
            "out": out,
            "prompt": "A house cat sits quietly in a living room.",
            "dry_run": True,
        }
        if record.input_kind == "image":
            compile_kwargs["image"] = dummy_image
        elif record.input_kind == "source_video":
            compile_kwargs["source_video"] = dummy_video

        try:
            result = wrbench.compile_camera(**compile_kwargs)
            artifacts = result.get("artifacts", {})
            written = [p for p in artifacts.values() if Path(p).exists()]
            lines.append(f"  pass  dry-run: compile succeeded, {len(written)} artifacts written")
        except Exception as exc:
            lines.append(f"  FAIL  dry-run: {exc}")
            return False, lines

    # Check 5 (optional): generation backend readiness when runtime config configures this model
    if runtime is not None:
        models_map = getattr(runtime, "models", {}) or {}
        if key in models_map:
            from wrbench.backends.registry import resolve_backend

            try:
                backend = resolve_backend(key, runtime=runtime)
                if hasattr(backend, "available_for"):
                    b_ok, b_reason = backend.available_for(key)  # type: ignore[attr-defined]
                else:
                    b_ok, b_reason = backend.available()
                if b_ok:
                    lines.append(f"  pass  backend: {backend.name} ({b_reason})")
                else:
                    lines.append(f"  FAIL  backend: {backend.name} ({b_reason})")
                    return False, lines
            except Exception as exc:
                lines.append(f"  FAIL  backend: {exc}")
                return False, lines
        else:
            lines.append("  info  backend: dry_run (not configured in runtime config)")

    lines.append(f"  PASS  {key}")
    return True, lines


def _cmd_doctor(args: argparse.Namespace) -> int:
    import wrbench

    runtime_path = Path(args.runtime_config) if args.runtime_config else None
    runtime_obj: Any = None
    runtime_load_error: str | None = None

    if runtime_path is not None:
        from wrbench.runtime import RuntimeConfigError, load_runtime_config

        try:
            runtime_obj = load_runtime_config(runtime_path)
        except (RuntimeConfigError, OSError, ValueError) as exc:
            runtime_load_error = str(exc)

    eval_only = bool(args.eval) and not args.model and not args.all
    run_eval_checks = bool(args.eval) or (
        runtime_path is not None and (bool(args.all) or _runtime_has_eval_block(runtime_path))
    )

    if runtime_load_error is not None and not eval_only:
        msg = f"failed to load runtime config {runtime_path}: {runtime_load_error}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, indent=2))
        else:
            print(f"error: {msg}", file=sys.stderr)
        return 1

    model_results: list[dict[str, Any]] = []
    overall_ok = True

    if not eval_only:
        if args.all:
            all_rec = wrbench.all_records()
            targets = [r.key for r in all_rec]
        elif args.model:
            try:
                targets = [wrbench.canonical_model_key(args.model)]
            except KeyError:
                known = wrbench.list_models(include_deferred=True)
                msg = f"unknown model {args.model!r}"
                if args.json:
                    print(json.dumps({"ok": False, "error": msg, "known_models": known}, indent=2))
                else:
                    print(f"error: {msg}", file=sys.stderr)
                    print(f"  known models: {', '.join(known)}", file=sys.stderr)
                return 1
        else:
            targets = wrbench.list_models(include_deferred=False)

        for key in targets:
            ok, lines = _doctor_check_model(key, runtime=runtime_obj)
            model_results.append({"key": key, "ok": ok, "lines": lines})
            if not ok:
                overall_ok = False
            if not args.json:
                print(f"\n[{key}]")
                for line in lines:
                    print(line)

        if not args.json and targets:
            passed = sum(1 for item in model_results if item["ok"])
            failed_keys = [item["key"] for item in model_results if not item["ok"]]
            print(f"\n{'=' * 60}")
            print(f"Summary: {passed}/{len(targets)} passed")
            if failed_keys:
                print(f"Failed:  {', '.join(failed_keys)}")

    eval_payload: dict[str, Any] | None = None
    if run_eval_checks:
        from wrbench.eval.doctor import check_eval_runtime, format_eval_doctor_report

        eval_ok, checks = check_eval_runtime(runtime_path)
        eval_payload = {"ok": eval_ok, "checks": checks}
        if not eval_ok:
            overall_ok = False
        if not args.json:
            print("\n[eval runtime]")
            print(format_eval_doctor_report(checks))

    if args.json:
        out: dict[str, Any] = {"ok": overall_ok}
        if not eval_only:
            out["models"] = model_results
        if eval_payload is not None:
            out["eval"] = eval_payload
        print(json.dumps(out, indent=2))
        return 0 if overall_ok else 1

    if overall_ok:
        print("All checks passed.")
        return 0
    return 1


def register_doctor_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the doctor subparser."""
    p_doc = sub.add_parser(
        "doctor", help="Validate registry, adapters, generation backends, and eval runtime."
    )
    model_group = p_doc.add_mutually_exclusive_group()
    model_group.add_argument(
        "--model", metavar="MODEL", help="Check a single model."
    )
    model_group.add_argument(
        "--all", action="store_true", help="Check all models including deferred."
    )
    p_doc.add_argument(
        "--runtime-config",
        dest="runtime_config",
        metavar="PATH",
        help="Optional wrbench.runtime.json path to check generation backend and/or eval runtime readiness.",
    )
    p_doc.add_argument(
        "--eval",
        action="store_true",
        help="Run evaluation runtime diagnostics (check scorer paths, custom backends, and Qwen/Transformers version).",
    )
    p_doc.add_argument(
        "--json",
        action="store_true",
        help="Output diagnostic results as JSON.",
    )
    add_model_spec_argument(p_doc)
