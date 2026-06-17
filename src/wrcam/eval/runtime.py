"""Eval runtime configuration and orchestration helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wrcam.runtime import RuntimeConfig, load_runtime_config


@dataclass(frozen=True)
class EvalScorerRuntime:
    """Paths to external model scorers (host-only, not pip deps)."""

    vggt_python_bin: str | None = None
    vggt_repo: str | None = None
    vggt_checkpoint: str | None = None
    dinov2_python_bin: str | None = None
    dinov2_model_path: str | None = None
    qwen_scorer_python: str | None = None
    qwen35_model: str | None = None
    qwen3vl_model: str | None = None
    gpu_id: int = 0
    extra_paths: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalRuntimeConfig:
    schema_version: int
    scorers: EvalScorerRuntime
    defaults: dict[str, Any] = field(default_factory=dict)


def _parse_eval_node(payload: dict[str, Any], defaults: dict[str, Any]) -> EvalScorerRuntime:
    eval_block = payload.get("eval")
    if isinstance(eval_block, dict) and isinstance(eval_block.get("scorers"), dict):
        node = eval_block["scorers"]
    elif isinstance(payload.get("scorers"), dict):
        node = payload["scorers"]
    else:
        node = {}
    return EvalScorerRuntime(
        vggt_python_bin=node.get("vggt_python_bin"),
        vggt_repo=node.get("vggt_repo"),
        vggt_checkpoint=node.get("vggt_checkpoint"),
        dinov2_python_bin=node.get("dinov2_python_bin"),
        dinov2_model_path=node.get("dinov2_model_path"),
        qwen_scorer_python=node.get("qwen_scorer_python"),
        qwen35_model=node.get("qwen35_model"),
        qwen3vl_model=node.get("qwen3vl_model"),
        gpu_id=int(node.get("gpu_id", defaults.get("gpu_id", 0))),
        extra_paths={str(k): str(v) for k, v in (node.get("extra_paths") or {}).items()},
        env={str(k): str(v) for k, v in (node.get("env") or {}).items()},
    )


def load_eval_runtime(path: Path | None = None) -> EvalRuntimeConfig | None:
    runtime = load_runtime_config(path)
    if runtime is None:
        return None
    resolved = path
    if resolved is None:
        from wrcam.runtime import _resolve_runtime_path

        resolved = _resolve_runtime_path(None)
    if resolved is None:
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    defaults = dict(payload.get("defaults") or {})
    if "eval" not in payload and "scorers" not in payload:
        return None
    return EvalRuntimeConfig(
        schema_version=int(payload.get("schema_version", runtime.schema_version)),
        scorers=_parse_eval_node(payload, defaults),
        defaults=defaults,
    )


def require_eval_runtime(path: Path | None = None) -> EvalRuntimeConfig:
    cfg = load_eval_runtime(path)
    if cfg is None:
        raise RuntimeError(
            "eval scorers not configured: add an 'eval.scorers' section to wrcam.runtime.json "
            "(see wrcam.runtime.example.json)"
        )
    return cfg


def contract_path() -> Path:
    return Path(__file__).resolve().parent / "contract" / "latest_d1_d6_metric_contract.json"


def run_module_main(module: str, argv: list[str], *, env: dict[str, str] | None = None) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    cmd = [sys.executable, "-m", module, *argv]
    proc = subprocess.run(cmd, env=merged)
    return int(proc.returncode)


def run_scorer_python(
    python_bin: str,
    module: str,
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
) -> int:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    cmd = [python_bin, "-m", module, *argv]
    proc = subprocess.run(cmd, env=merged)
    return int(proc.returncode)


def run_shell_script(script: Path, stage: str, *, env: dict[str, str]) -> int:
    merged = os.environ.copy()
    merged.update(env)
    proc = subprocess.run(["bash", str(script), stage], env=merged)
    return int(proc.returncode)


def d1_score(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    summary_csv: Path,
    pose_cache_root: Path,
    pose_backend: str = "vggt_omega",
    poses_file: str = "poses.npy",
    default_frames: int = 121,
    sidecar_profile_gate: str = "main",
) -> int:
    from wrcam.eval.d1.d1_camera import main as d1_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-jsonl",
        str(output_jsonl),
        "--summary-csv",
        str(summary_csv),
        "--megasam-cache-root",
        str(pose_cache_root),
        "--pose-backend",
        pose_backend,
        "--poses-file",
        poses_file,
        "--default-frames",
        str(default_frames),
        "--sidecar-profile-gate",
        sidecar_profile_gate,
    ]
    return d1_main(argv)


def d1_camalign_score(
    *,
    input_jsonl: Path,
    output_jsonl: Path,
    pose_cache_root: Path,
    poses_file: str = "poses.npy",
) -> int:
    from wrcam.eval.d1.d1_camalign import main as camalign_main

    return camalign_main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--output-jsonl",
            str(output_jsonl),
            "--pose-cache-root",
            str(pose_cache_root),
            "--poses-file",
            poses_file,
        ]
    )


def d1_vggt_batch(
    *,
    eval_runtime: EvalRuntimeConfig,
    input_jsonl: Path,
    output_root: Path,
    cache_root: Path | None = None,
    execution_mode: str = "subprocess",
) -> int:
    scorers = eval_runtime.scorers
    for field_name, value in (
        ("vggt_python_bin", scorers.vggt_python_bin),
        ("vggt_repo", scorers.vggt_repo),
        ("vggt_checkpoint", scorers.vggt_checkpoint),
    ):
        if not value or not Path(str(value)).exists():
            raise FileNotFoundError(f"eval.scorers.{field_name} missing or not found: {value!r}")
    from wrcam.eval.d1.run_d1_vggt_omega_batch import main as batch_main

    argv = [
        "--input-jsonl",
        str(input_jsonl),
        "--output-root",
        str(output_root),
        "--vggt-repo",
        str(scorers.vggt_repo),
        "--checkpoint",
        str(scorers.vggt_checkpoint),
        "--gpu-id",
        str(scorers.gpu_id),
        "--python",
        str(scorers.vggt_python_bin or sys.executable),
        "--execution-mode",
        execution_mode,
        "--poses-file",
        "poses.npy",
    ]
    if cache_root is not None:
        argv.extend(["--cache-root", str(cache_root)])
    return batch_main(argv)


def d2_extract(
    *,
    eval_runtime: EvalRuntimeConfig,
    videos_manifest: Path,
    out_jsonl: Path,
    model_dir: Path | None = None,
) -> int:
    scorers = eval_runtime.scorers
    python_bin = scorers.dinov2_python_bin or sys.executable
    model = model_dir or (Path(scorers.dinov2_model_path) if scorers.dinov2_model_path else None)
    argv = [
        "--videos",
        str(videos_manifest),
        "--out-jsonl",
        str(out_jsonl),
    ]
    if model is not None:
        argv.extend(["--model-dir", str(model)])
    return run_scorer_python(
        python_bin,
        "wrcam.eval.d2.extract_d2_dinov2_local_global_candidate",
        argv,
        env={"CUDA_VISIBLE_DEVICES": str(scorers.gpu_id), **scorers.env},
    )


def wrcam_repo_root() -> Path:
    """Return the WRCam repository root (directory containing pyproject.toml)."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here.parents[3]


def d3d6_env(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scorer_profile: str = "wrbench_default",
) -> dict[str, str]:
    profile = normalize_scorer_profile(scorer_profile)
    scorers = eval_runtime.scorers
    repo_root = wrcam_repo_root()
    shell = repo_root / "scripts" / "eval" / "score_runtime_v2_d3d6.sh"
    env = {
        "MANIFEST": str(manifest.resolve()),
        "OUT_DIR": str(out_dir.resolve()),
        "SCORER_PROFILE": profile,
        "PY_SCORER": scorers.qwen_scorer_python or sys.executable,
        "PY_HELPER": scorers.qwen_scorer_python or sys.executable,
        "QWEN35_MODEL": scorers.qwen35_model or "",
        "QWEN3VL_MODEL": scorers.qwen3vl_model or "",
        "CUDA_VISIBLE_DEVICES": str(scorers.gpu_id),
        "PYTHONPATH": str(repo_root),
        **scorers.env,
    }
    if not shell.is_file():
        raise FileNotFoundError(f"D3-D6 shell orchestrator missing: {shell}")
    env["WRCAM_D3D6_SCRIPT"] = str(shell)
    return env


def d3d6_score(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    stage: str = "all",
    scorer_profile: str = "wrbench_default",
) -> int:
    env = d3d6_env(
        eval_runtime=eval_runtime,
        manifest=manifest,
        out_dir=out_dir,
        scorer_profile=scorer_profile,
    )
    script = Path(env["WRCAM_D3D6_SCRIPT"])
    return run_shell_script(script, stage, env=env)


def normalize_scorer_profile(profile: str) -> str:
    aliases = {
        "current_benchmark_p25_p22_e14": "wrbench_default",
        "legacy_p9_all_manifest_metadata": "ablation_manifest_metadata",
    }
    return aliases.get(profile, profile)


def eval_run(
    *,
    eval_runtime: EvalRuntimeConfig,
    manifest: Path,
    out_dir: Path,
    scorer_profile: str = "wrbench_default",
    sidecar_profile_gate: str = "main",
) -> int:
    """Run the full WRBench eval pipeline: D1 pose -> D1 score -> D2 -> D3-D6 -> table."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = manifest.resolve()
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("manifest must be a JSON list")

    d1_input = out_dir / "d1_input.jsonl"
    d1_input.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows if isinstance(row, dict)),
        encoding="utf-8",
    )
    pose_root = out_dir / "d1_vggt"
    cache_root = pose_root / "cache"
    d1_scored = out_dir / "d1_scored.jsonl"
    d1_camalign_scored = out_dir / "d1_camalign_scored.jsonl"
    d1_summary = out_dir / "d1_summary.csv"
    d2_out = out_dir / "d2_features.jsonl"
    d3d6_out = out_dir / "d3d6"
    runtime_scores = d3d6_out / "final_exports" / "scores_v7_runtime_v2_evidence_first_gate_masked_export.json"

    steps: list[tuple[str, int]] = []
    rc = d1_vggt_batch(
        eval_runtime=eval_runtime,
        input_jsonl=d1_input,
        output_root=pose_root,
        cache_root=cache_root,
    )
    steps.append(("d1-vggt", rc))
    if rc != 0:
        return rc

    rc = d1_score(
        input_jsonl=d1_input,
        output_jsonl=d1_scored,
        summary_csv=d1_summary,
        pose_cache_root=cache_root,
        sidecar_profile_gate=sidecar_profile_gate,
    )
    steps.append(("d1", rc))
    if rc != 0:
        return rc

    rc = d1_camalign_score(
        input_jsonl=d1_input,
        output_jsonl=d1_camalign_scored,
        pose_cache_root=cache_root,
    )
    steps.append(("d1-camalign", rc))
    if rc != 0:
        return rc

    rc = d2_extract(
        eval_runtime=eval_runtime,
        videos_manifest=manifest,
        out_jsonl=d2_out,
    )
    steps.append(("d2", rc))
    if rc != 0:
        return rc

    rc = d3d6_score(
        eval_runtime=eval_runtime,
        manifest=manifest,
        out_dir=d3d6_out,
        stage="all",
        scorer_profile=scorer_profile,
    )
    steps.append(("d3d6", rc))
    if rc != 0:
        return rc

    if not runtime_scores.is_file():
        raise FileNotFoundError(f"D3-D6 export missing: {runtime_scores}")

    rc = build_table(
        [
            "--runtime-scores",
            str(runtime_scores),
            "--d1-scores",
            str(d1_scored),
            "--d1-camalign-scores",
            str(d1_camalign_scored),
            "--d2-scores",
            str(d2_out),
            "--out-csv",
            str(out_dir / "main_table.csv"),
            "--out-md",
            str(out_dir / "main_table.md"),
            "--out-summary",
            str(out_dir / "main_table_summary.json"),
        ]
    )
    steps.append(("table", rc))
    return rc


def build_table(argv: list[str] | None = None) -> int:
    from wrcam.eval.aggregate.build_wrbench_vnext_main_table import main as table_main

    return table_main(argv)
