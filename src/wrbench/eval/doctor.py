"""Diagnostics for WRBench evaluation runtime configuration and scorer environments."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any

from wrbench.eval.plugins import DEFAULT_BACKENDS, get_backend
from wrbench.eval.runtime import load_eval_runtime
from wrbench.eval.scoring.runtime_common import HISTORICAL_TRANSFORMERS_PREFIX
from wrbench.runtime import RuntimeConfigError

_PROBE_SCRIPT = (
    "import json\n"
    "def _v(name):\n"
    "    try:\n"
    "        m = __import__(name)\n"
    "    except Exception:\n"
    "        return 'unavailable'\n"
    "    return str(getattr(m, '__version__', 'unknown'))\n"
    "print(json.dumps({"
    "'transformers': _v('transformers'), "
    "'torch': _v('torch'), "
    "'tokenizers': _v('tokenizers'), "
    "'decord': _v('decord')"
    "}))\n"
)


def _check(
    *,
    name: str,
    stage: str,
    status: str,
    message: str,
    path: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "stage": stage,
        "status": status,
        "message": message,
        "path": path,
        "details": dict(details or {}),
    }


def _check_path(
    *,
    name: str,
    stage: str,
    value: str | None,
    kind: str,
    field_label: str,
) -> dict[str, Any]:
    if not value or not str(value).strip():
        return _check(
            name=name,
            stage=stage,
            status="error",
            message=f"{field_label} is not configured",
            path=None,
        )
    resolved = Path(str(value)).expanduser()
    if kind == "dir":
        if resolved.is_dir():
            return _check(
                name=name,
                stage=stage,
                status="ok",
                message=f"{field_label} directory exists",
                path=str(resolved),
            )
        return _check(
            name=name,
            stage=stage,
            status="error",
            message=f"{field_label} directory not found: {resolved}",
            path=str(resolved),
        )
    if kind == "file":
        if resolved.is_file():
            return _check(
                name=name,
                stage=stage,
                status="ok",
                message=f"{field_label} file exists",
                path=str(resolved),
            )
        return _check(
            name=name,
            stage=stage,
            status="error",
            message=f"{field_label} file not found: {resolved}",
            path=str(resolved),
        )
    if resolved.exists():
        return _check(
            name=name,
            stage=stage,
            status="ok",
            message=f"{field_label} path exists",
            path=str(resolved),
        )
    return _check(
        name=name,
        stage=stage,
        status="error",
        message=f"{field_label} path not found: {resolved}",
        path=str(resolved),
    )


def _probe_qwen_python(python_path: Path) -> list[dict[str, Any]]:
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        return [
            _check(
                name="vlm.environment",
                stage="vlm",
                status="warn",
                message=f"Skipping environment probe: {python_path} is not executable",
                path=str(python_path),
            )
        ]
    try:
        proc = subprocess.run(
            [str(python_path), "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as exc:
        return [
            _check(
                name="vlm.environment",
                stage="vlm",
                status="warn",
                message=f"Failed to probe qwen_scorer_python ({python_path}): {exc}",
                path=str(python_path),
            )
        ]

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip().splitlines()
        hint = stderr[-1] if stderr else f"exit code {proc.returncode}"
        return [
            _check(
                name="vlm.environment",
                stage="vlm",
                status="warn",
                message=f"qwen_scorer_python probe returned non-zero ({hint})",
                path=str(python_path),
            )
        ]

    try:
        versions = json.loads((proc.stdout or "").strip())
        if not isinstance(versions, dict):
            raise ValueError("probe output is not a JSON object")
    except Exception as exc:
        return [
            _check(
                name="vlm.environment",
                stage="vlm",
                status="warn",
                message=f"Unable to parse qwen_scorer_python probe output: {exc}",
                path=str(python_path),
            )
        ]

    tf_ver = str(versions.get("transformers", "unavailable"))
    torch_ver = str(versions.get("torch", "unavailable"))
    tok_ver = str(versions.get("tokenizers", "unavailable"))
    decord_ver = str(versions.get("decord", "unavailable"))

    results: list[dict[str, Any]] = [
        _check(
            name="vlm.environment",
            stage="vlm",
            status="ok",
            message=(
                f"Probed qwen_scorer_python: transformers={tf_ver}, "
                f"torch={torch_ver}, tokenizers={tok_ver}, decord={decord_ver}"
            ),
            path=str(python_path),
            details={
                "transformers": tf_ver,
                "torch": torch_ver,
                "tokenizers": tok_ver,
                "decord": decord_ver,
            },
        )
    ]

    if tf_ver in ("unavailable", "unknown", ""):
        results.append(
            _check(
                name="vlm.transformers_version",
                stage="vlm",
                status="warn",
                message=(
                    "transformers is not importable in qwen_scorer_python. "
                    "Install transformers==5.9.0 for historical D3/D4 replay (see docs/eval/hunyuan-d3-replay.md)."
                ),
                path=str(python_path),
                details={"transformers": tf_ver},
            )
        )
    elif not tf_ver.startswith(HISTORICAL_TRANSFORMERS_PREFIX):
        results.append(
            _check(
                name="vlm.transformers_version",
                stage="vlm",
                status="warn",
                message=(
                    f"qwen_scorer_python has transformers=={tf_ver} (historical replay baseline is 5.9.0). "
                    "Newer Transformers processors (>=5.22) retain extra <|vision_start|>/<|vision_end|> "
                    "boundary tokens around expanded video pads, shifting multimodal RoPE by 2 and lowering "
                    "D3/D4 scores (Issue #11; see docs/eval/hunyuan-d3-replay.md). Set "
                    "QWEN35_VISION_BOUNDARY_MODE=strip_outer_markers or install transformers==5.9.0."
                ),
                path=str(python_path),
                details={"transformers": tf_ver, "expected_prefix": HISTORICAL_TRANSFORMERS_PREFIX},
            )
        )
    else:
        results.append(
            _check(
                name="vlm.transformers_version",
                stage="vlm",
                status="ok",
                message=f"transformers=={tf_ver} matches historical 5.9.x replay baseline",
                path=str(python_path),
                details={"transformers": tf_ver},
            )
        )

    if torch_ver == "unavailable":
        results.append(
            _check(
                name="vlm.torch_version",
                stage="vlm",
                status="warn",
                message="torch is not importable in qwen_scorer_python",
                path=str(python_path),
                details={"torch": torch_ver},
            )
        )
    if decord_ver == "unavailable":
        results.append(
            _check(
                name="vlm.decord_version",
                stage="vlm",
                status="warn",
                message=(
                    "decord is not importable in qwen_scorer_python "
                    "(default WORLD_STATE_VIDEO_BACKEND=decord requires decord)"
                ),
                path=str(python_path),
                details={"decord": decord_ver},
            )
        )
    return results


def check_eval_runtime(runtime_config: str | Path | None) -> tuple[bool, list[dict[str, Any]]]:
    """Validate eval runtime configuration, scorer paths, custom backends, and Qwen versions."""
    checks: list[dict[str, Any]] = []
    if runtime_config is None:
        checks.append(
            _check(
                name="runtime_config",
                stage="config",
                status="error",
                message=(
                    "No runtime config provided. Pass --runtime-config <path> containing an "
                    "'eval.scorers' or 'eval.backends' block."
                ),
            )
        )
        return False, checks

    config_path = Path(runtime_config)
    try:
        cfg = load_eval_runtime(config_path)
    except (FileNotFoundError, RuntimeConfigError, json.JSONDecodeError, ValueError) as exc:
        checks.append(
            _check(
                name="runtime_config",
                stage="config",
                status="error",
                message=str(exc),
                path=str(config_path),
            )
        )
        return False, checks

    if cfg is None:
        checks.append(
            _check(
                name="runtime_config",
                stage="config",
                status="error",
                message=f"{config_path}: missing 'eval' configuration block",
                path=str(config_path),
            )
        )
        return False, checks

    checks.append(
        _check(
            name="runtime_config",
            stage="config",
            status="ok",
            message=f"Loaded eval runtime config (schema_version={cfg.schema_version}, gpu_id={cfg.scorers.gpu_id})",
            path=str(config_path),
            details={"schema_version": cfg.schema_version, "gpu_id": cfg.scorers.gpu_id},
        )
    )

    scorers = cfg.scorers
    stage_checks = {
        "pose": (
            ("pose.vggt_python_bin", scorers.vggt_python_bin, "file", "eval.scorers.vggt_python_bin"),
            ("pose.vggt_repo", scorers.vggt_repo, "dir", "eval.scorers.vggt_repo"),
            ("pose.vggt_checkpoint", scorers.vggt_checkpoint, "exists", "eval.scorers.vggt_checkpoint"),
        ),
        "d2": (
            ("d2.dinov2_python_bin", scorers.dinov2_python_bin, "file", "eval.scorers.dinov2_python_bin"),
            ("d2.dinov2_model_path", scorers.dinov2_model_path, "exists", "eval.scorers.dinov2_model_path"),
        ),
        "vlm": (
            ("vlm.qwen_scorer_python", scorers.qwen_scorer_python, "file", "eval.scorers.qwen_scorer_python"),
            ("vlm.qwen35_model", scorers.qwen35_model, "dir", "eval.scorers.qwen35_model"),
            ("vlm.qwen3vl_model", scorers.qwen3vl_model, "dir", "eval.scorers.qwen3vl_model"),
        ),
    }

    for stage in ("pose", "d2", "vlm"):
        selected = cfg.backends.get(stage)
        is_custom = bool(selected and selected != DEFAULT_BACKENDS[stage])
        if is_custom:
            try:
                get_backend(stage, selected)
                checks.append(
                    _check(
                        name=f"{stage}.backend",
                        stage=stage,
                        status="ok",
                        message=(
                            f"Custom {stage} backend {selected!r} resolved cleanly "
                            f"(built-in {stage} scorer paths not required)"
                        ),
                        details={"backend": selected},
                    )
                )
            except Exception as exc:
                checks.append(
                    _check(
                        name=f"{stage}.backend",
                        stage=stage,
                        status="error",
                        message=f"Custom {stage} backend {selected!r} failed to load: {exc}",
                        details={"backend": selected},
                    )
                )
            continue

        for name, value, kind, label in stage_checks[stage]:
            checks.append(_check_path(name=name, stage=stage, value=value, kind=kind, field_label=label))

        if stage == "vlm" and scorers.qwen_scorer_python:
            py_path = Path(str(scorers.qwen_scorer_python)).expanduser()
            if py_path.is_file():
                checks.extend(_probe_qwen_python(py_path))

    ok = all(item["status"] != "error" for item in checks)
    return ok, checks


def format_eval_doctor_report(checks: list[dict[str, Any]]) -> str:
    """Render ``check_eval_runtime`` checks as a concise human-readable report."""
    lines = ["WRBench Eval Doctor"]
    for item in checks:
        status = str(item.get("status", "ok")).upper()
        name = str(item.get("name", ""))
        message = str(item.get("message", ""))
        lines.append(f"[{status}] {name}: {message}")
    errors = sum(1 for item in checks if item.get("status") == "error")
    warnings = sum(1 for item in checks if item.get("status") == "warn")
    verdict = "PASS" if errors == 0 else "FAIL"
    lines.append(f"Summary: {verdict} ({len(checks)} checks, {warnings} warnings, {errors} errors)")
    return "\n".join(lines)
