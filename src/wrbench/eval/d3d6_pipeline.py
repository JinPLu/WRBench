#!/usr/bin/env python3
"""Pure-Python orchestrator for the WRBench D3-D6 Qwen scoring pipeline.

Replaces the bash loop in ``score_runtime_v2_d3d6.sh`` so D3-D6 evaluation
runs natively without requiring ``bash`` while preserving all stage names,
CLI flags, shard layouts, log paths, and export contracts 1:1.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping, Sequence

from wrbench.eval._io import resolve_video_path

STAGES: tuple[str, ...] = (
    "preflight",
    "qwen35",
    "merge_qwen35",
    "gate_binary",
    "merge_binary",
    "build_rescue",
    "gate_rescue",
    "merge_rescue",
    "overlay_gate",
    "export",
    "all",
)

ALL_STAGE_ORDER: tuple[str, ...] = (
    "preflight",
    "qwen35",
    "merge_qwen35",
    "gate_binary",
    "merge_binary",
    "build_rescue",
    "gate_rescue",
    "merge_rescue",
    "overlay_gate",
    "export",
)

REQUIRED_ENV_VARS: tuple[str, ...] = (
    "FORCE_QWENVL_VIDEO_READER",
    "WORLD_STATE_VIDEO_BACKEND",
    "PY_SCORER",
    "PY_HELPER",
    "MANIFEST",
    "OUT_DIR",
    "QWEN35_MODEL",
    "QWEN3VL_MODEL",
    "SCORER_PROFILE",
    "RUN_TAG",
    "PROMPT_MODE",
    "TASK_CONTEXT_MODE",
    "NUM_SHARDS",
    "SHARD_IDS",
    "CUDA_DEVICES_CSV",
    "FPS",
    "PROGRESS_EVERY",
    "SKIP_EXISTING",
    "QWEN35_VLM_NAME",
    "QWEN35_LOADER_FAMILY",
    "QWEN35_DTYPE",
    "QWEN35_ATTN_IMPLEMENTATION",
    "QWEN35_NUM_SAMPLES",
    "QWEN35_MAX_VIDEOS",
    "QWEN35_EVIDENCE_CONTEXT_MODE",
    "QWEN3VL_DTYPE",
    "QWEN3VL_ATTN_IMPLEMENTATION",
    "QWEN3VL_MAX_NEW_TOKENS",
    "QWEN3VL_MAX_VIDEOS",
    "QWEN3VL_BINARY_PROMPT_SCHEMA",
    "QWEN3VL_RESCUE_PROMPT_SCHEMA",
)

DEFAULT_PROFILE_REQUIREMENTS: dict[str, str | tuple[str, ...]] = {
    "PROMPT_MODE": ("wrbench_paper", "runtime_v2_probe_logprob_p25_d3d4_slot_parse"),
    "TASK_CONTEXT_MODE": "none",
    "QWEN3VL_BINARY_PROMPT_SCHEMA": "subject_judgeability_v1",
    "QWEN3VL_RESCUE_PROMPT_SCHEMA": "guarded_teacher_gate_v3",
}

SUPPORTED_PROFILES: frozenset[str] = frozenset(
    {"default", "wrbench_default", "ablation_manifest_metadata", "custom"}
)

USAGE = """\
Usage:
  MANIFEST=/abs/path/manifest.json OUT_DIR=/abs/path/out \\
  QWEN35_MODEL=/abs/path/Qwen3.5-9B QWEN3VL_MODEL=/abs/path/Qwen3-VL-8B-Instruct \\
  python3 -m wrbench.eval.d3d6_pipeline <stage>

Stages:
  preflight | qwen35 | merge_qwen35 | gate_binary | merge_binary |
  build_rescue | gate_rescue | merge_rescue | overlay_gate | export | all

Manifest contract:
  JSON list. Each row must contain video_id, path or video_path, and
  world_state_prompt or prompt_text. Optional benchmark metadata fields are
  passed through when present.
"""


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _child_env(env: Mapping[str, str], *, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    merged = os.environ.copy()
    for key, value in env.items():
        merged[str(key)] = str(value)
    if extra:
        for key, value in extra.items():
            merged[str(key)] = str(value)
    package_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = merged.get("PYTHONPATH", "")
    parts = [p for p in (package_root, existing_pythonpath) if p]
    merged["PYTHONPATH"] = os.pathsep.join(parts)
    return merged


def _validate_env(stage: str, env: Mapping[str, str]) -> tuple[int, dict[str, str]]:
    resolved: dict[str, str] = {}
    for name in REQUIRED_ENV_VARS:
        value = str(env.get(name, "") or "")
        if not value:
            _err(f"[error] required environment variable {name} is not set")
            return 2, {}
        resolved[name] = value

    resolved["SOURCE_SCORES"] = str(env.get("SOURCE_SCORES", "") or "")
    resolved["QWEN35_VISION_BOUNDARY_MODE"] = str(
        env.get("QWEN35_VISION_BOUNDARY_MODE", "warn") or "warn"
    )

    if resolved["SKIP_EXISTING"] not in {"0", "1"}:
        _err("[error] SKIP_EXISTING must be 0 or 1")
        return 2, {}

    profile = resolved["SCORER_PROFILE"]
    if profile == "default":
        profile = "wrbench_default"
        resolved["SCORER_PROFILE"] = profile
    if profile not in SUPPORTED_PROFILES:
        _err(f"[error] unsupported SCORER_PROFILE: {profile}")
        _err("[error] expected default, wrbench_default, ablation_manifest_metadata, or custom")
        return 2, {}

    if profile == "wrbench_default":
        for key, expected in DEFAULT_PROFILE_REQUIREMENTS.items():
            allowed = expected if isinstance(expected, tuple) else (expected,)
            if resolved[key] not in allowed:
                expected_display = " or ".join(allowed)
                _err(f"[error] wrbench_default requires {key}={expected_display}")
                return 2, {}
        if not re.fullmatch(r"2(\.0+)?", resolved["FPS"]):
            _err("[error] wrbench_default requires FPS=2")
            return 2, {}

    if stage not in STAGES:
        _err(f"[error] unknown stage: {stage}")
        _err(USAGE)
        return 2, {}

    manifest = resolved["MANIFEST"]
    if not manifest:
        _err("[error] MANIFEST is required")
        _err(USAGE)
        return 2, {}
    if not Path(manifest).is_file():
        _err(f"[error] MANIFEST is not readable: {manifest}")
        return 2, {}

    return 0, resolved


def _preflight(manifest_path: Path) -> int:
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        _err("manifest must be a JSON list")
        return 1
    seen: set[str] = set()
    missing: list[int] = []
    unreadable: list[str] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            _err(f"manifest row {idx} is not an object")
            return 1
        video_id = str(row.get("video_id") or "").strip()
        video_path = resolve_video_path(row, prefer_eval_video=False).strip()
        prompt = str(row.get("world_state_prompt") or row.get("prompt_text") or "").strip()
        if not video_id or not video_path or not prompt:
            missing.append(idx)
        if video_id in seen:
            _err(f"duplicate video_id: {video_id}")
            return 1
        seen.add(video_id)
        if video_path and not Path(video_path).is_file():
            unreadable.append(video_id or f"row-{idx}")
    if missing:
        _err(f"rows missing video_id/path/world_state_prompt: {missing[:10]}")
        return 1
    if unreadable:
        _err(f"unreadable video paths: {unreadable[:10]}")
        return 1
    print(f"[preflight] manifest_records={len(rows)}", flush=True)
    return 0


def _json_list_len(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data)


def _run_command(cmd: Sequence[str], *, env: Mapping[str, str]) -> int:
    proc = subprocess.run(list(cmd), env=dict(env))
    return int(proc.returncode)


def _run_sharded(
    *,
    name: str,
    py: str,
    module_args: Sequence[str],
    num_shards: str,
    shard_ids_raw: str,
    cuda_devices_csv: str,
    log_dir: Path,
    base_env: Mapping[str, str],
) -> int:
    gpus = [g.strip() for g in cuda_devices_csv.split(",") if g.strip()]
    if not gpus:
        _err("[error] CUDA_DEVICES_CSV must contain at least one device")
        return 2
    shard_ids = [s for s in re.split(r"[\s,]+", shard_ids_raw.strip()) if s]
    if not shard_ids:
        _err("[error] SHARD_IDS must contain at least one shard id")
        return 2

    procs: list[tuple[subprocess.Popen[bytes], Any]] = []
    for idx, sid in enumerate(shard_ids):
        gpu = gpus[idx % len(gpus)]
        print(f"[launch] {name} shard={sid}/{num_shards} gpu={gpu}", flush=True)
        log_path = log_dir / f"{name}_shard_{sid}.log"
        log_file = log_path.open("wb")
        shard_env = dict(base_env)
        shard_env["CUDA_VISIBLE_DEVICES"] = gpu
        cmd = [
            py,
            *module_args,
            "--num-shards",
            str(num_shards),
            "--shard-id",
            str(sid),
            "--local-rank",
            "0",
        ]
        proc = subprocess.Popen(cmd, env=shard_env, stdout=log_file, stderr=subprocess.STDOUT)
        procs.append((proc, log_file))

    status = 0
    for proc, log_file in procs:
        try:
            rc = proc.wait()
            if rc != 0:
                status = 1
        finally:
            log_file.close()
    return status


def run_d3d6_pipeline(*, stage: str, env: Mapping[str, str]) -> int:
    """Execute one or all stages of the WRBench D3-D6 evaluation pipeline."""
    rc, cfg = _validate_env(stage, env)
    if rc != 0:
        return rc

    manifest = Path(cfg["MANIFEST"])
    out_dir = Path(cfg["OUT_DIR"])
    qwen35_out = out_dir / "qwen35_worldstate_scores"
    binary_gate_out = out_dir / "qwen3vl_oov_binary_gate"
    rescue_out = out_dir / "qwen3vl_oov_guarded_rescue"
    gate_overlay_out = out_dir / "qwen3vl_oov_gate_overlay"
    final_out = out_dir / "final_exports"
    log_dir = out_dir / "logs"

    for directory in (qwen35_out, binary_gate_out, rescue_out, gate_overlay_out, final_out, log_dir):
        directory.mkdir(parents=True, exist_ok=True)

    base_env = _child_env(env, extra={
        "FORCE_QWENVL_VIDEO_READER": cfg["FORCE_QWENVL_VIDEO_READER"],
        "WORLD_STATE_VIDEO_BACKEND": cfg["WORLD_STATE_VIDEO_BACKEND"],
        "QWEN35_VISION_BOUNDARY_MODE": cfg["QWEN35_VISION_BOUNDARY_MODE"],
    })

    py_scorer = cfg["PY_SCORER"]
    py_helper = cfg["PY_HELPER"]
    skip_existing = cfg["SKIP_EXISTING"] == "1"

    qwen35_base_args: list[str] = [
        "-m",
        "wrbench.eval.scoring.run_local_qwen35_probe_logprob_scorer",
        "--experiment-id",
        cfg["RUN_TAG"],
        "--manifest-path",
        str(manifest),
        "--output-dir",
        str(qwen35_out),
        "--model-path",
        cfg["QWEN35_MODEL"],
        "--fps",
        cfg["FPS"],
        "--max-videos",
        cfg["QWEN35_MAX_VIDEOS"],
        "--progress-every",
        cfg["PROGRESS_EVERY"],
        "--prompt-mode",
        cfg["PROMPT_MODE"],
        "--task-context-mode",
        cfg["TASK_CONTEXT_MODE"],
        "--vlm-name",
        cfg["QWEN35_VLM_NAME"],
        "--loader-family",
        cfg["QWEN35_LOADER_FAMILY"],
        "--dtype",
        cfg["QWEN35_DTYPE"],
        "--attn-implementation",
        cfg["QWEN35_ATTN_IMPLEMENTATION"],
        "--num-samples",
        cfg["QWEN35_NUM_SAMPLES"],
        "--evidence-context-mode",
        cfg["QWEN35_EVIDENCE_CONTEXT_MODE"],
    ]
    if cfg["SOURCE_SCORES"]:
        qwen35_base_args.extend(["--source-scores", cfg["SOURCE_SCORES"]])
    if skip_existing:
        qwen35_base_args.append("--skip-existing")
    else:
        qwen35_base_args.append("--strict-manifest-contract")

    maybe_skip: list[str] = ["--skip-existing"] if skip_existing else []

    def _qwen3vl_args(manifest_arg: Path, output_arg: Path, prompt_schema: str) -> list[str]:
        return [
            "-m",
            "wrbench.eval.scoring.run_local_qwen3vl_video_evidence",
            "--manifest-path",
            str(manifest_arg),
            "--output-dir",
            str(output_arg),
            "--model-path",
            cfg["QWEN3VL_MODEL"],
            "--prompt-schema",
            prompt_schema,
            "--fps",
            cfg["FPS"],
            "--max-videos",
            cfg["QWEN3VL_MAX_VIDEOS"],
            "--dtype",
            cfg["QWEN3VL_DTYPE"],
            "--attn-implementation",
            cfg["QWEN3VL_ATTN_IMPLEMENTATION"],
            "--max-new-tokens",
            cfg["QWEN3VL_MAX_NEW_TOKENS"],
        ]

    def _run_stage(selected: str) -> int:
        if selected == "preflight":
            return _preflight(manifest)
        if selected == "qwen35":
            if not cfg["QWEN35_MODEL"] or not Path(cfg["QWEN35_MODEL"]).is_dir():
                _err(
                    "QWEN35_MODEL must point to a local model directory "
                    "(configure eval.scorers.qwen35_model)"
                )
                return 1
            return _run_sharded(
                name="qwen35",
                py=py_scorer,
                module_args=qwen35_base_args,
                num_shards=cfg["NUM_SHARDS"],
                shard_ids_raw=cfg["SHARD_IDS"],
                cuda_devices_csv=cfg["CUDA_DEVICES_CSV"],
                log_dir=log_dir,
                base_env=base_env,
            )
        if selected == "merge_qwen35":
            return _run_command(
                [
                    py_scorer,
                    *qwen35_base_args,
                    "--num-shards",
                    cfg["NUM_SHARDS"],
                    "--shard-id",
                    "0",
                    "--local-rank",
                    "0",
                    "--merge-only",
                ],
                env=base_env,
            )
        if selected == "gate_binary":
            return _run_sharded(
                name="qwen3vl_binary",
                py=py_scorer,
                module_args=[
                    *_qwen3vl_args(manifest, binary_gate_out, cfg["QWEN3VL_BINARY_PROMPT_SCHEMA"]),
                    *maybe_skip,
                ],
                num_shards=cfg["NUM_SHARDS"],
                shard_ids_raw=cfg["SHARD_IDS"],
                cuda_devices_csv=cfg["CUDA_DEVICES_CSV"],
                log_dir=log_dir,
                base_env=base_env,
            )
        if selected == "merge_binary":
            return _run_command(
                [
                    py_scorer,
                    *_qwen3vl_args(manifest, binary_gate_out, cfg["QWEN3VL_BINARY_PROMPT_SCHEMA"]),
                    "--num-shards",
                    cfg["NUM_SHARDS"],
                    "--shard-id",
                    "0",
                    "--local-rank",
                    "0",
                    "--merge-only",
                ],
                env=base_env,
            )
        if selected == "build_rescue":
            return _run_command(
                [
                    py_helper,
                    "-m",
                    "wrbench.eval.scoring.build_qwen3vl_binary_na_rescue_manifest",
                    "--manifest-path",
                    str(manifest),
                    "--binary-evidence-jsonl",
                    str(binary_gate_out / "evidence.jsonl"),
                    "--out-manifest",
                    str(rescue_out / "manifest_qwen3vl_guarded_rescue.json"),
                    "--out-summary",
                    str(rescue_out / "manifest_qwen3vl_guarded_rescue_summary.json"),
                ],
                env=base_env,
            )
        if selected == "gate_rescue":
            rescue_manifest = rescue_out / "manifest_qwen3vl_guarded_rescue.json"
            if _json_list_len(rescue_manifest) == 0:
                print("[skip] qwen3vl_rescue: rescue manifest is empty", flush=True)
                return 0
            return _run_sharded(
                name="qwen3vl_rescue",
                py=py_scorer,
                module_args=[
                    *_qwen3vl_args(rescue_manifest, rescue_out, cfg["QWEN3VL_RESCUE_PROMPT_SCHEMA"]),
                    *maybe_skip,
                ],
                num_shards=cfg["NUM_SHARDS"],
                shard_ids_raw=cfg["SHARD_IDS"],
                cuda_devices_csv=cfg["CUDA_DEVICES_CSV"],
                log_dir=log_dir,
                base_env=base_env,
            )
        if selected == "merge_rescue":
            rescue_manifest = rescue_out / "manifest_qwen3vl_guarded_rescue.json"
            if _json_list_len(rescue_manifest) == 0:
                (rescue_out / "evidence.jsonl").write_text("", encoding="utf-8")
                print("[skip] qwen3vl_rescue merge: rescue manifest is empty", flush=True)
                return 0
            return _run_command(
                [
                    py_scorer,
                    *_qwen3vl_args(rescue_manifest, rescue_out, cfg["QWEN3VL_RESCUE_PROMPT_SCHEMA"]),
                    "--num-shards",
                    cfg["NUM_SHARDS"],
                    "--shard-id",
                    "0",
                    "--local-rank",
                    "0",
                    "--merge-only",
                ],
                env=base_env,
            )
        if selected == "overlay_gate":
            return _run_command(
                [
                    py_helper,
                    "-m",
                    "wrbench.eval.scoring.merge_qwen3vl_rescue_evidence",
                    "--manifest-path",
                    str(manifest),
                    "--baseline-evidence-jsonl",
                    str(binary_gate_out / "evidence.jsonl"),
                    "--rescue-evidence-jsonl",
                    str(rescue_out / "evidence.jsonl"),
                    "--out-evidence-jsonl",
                    str(gate_overlay_out / "evidence_qwen3vl_guarded_overlay.jsonl"),
                    "--out-summary-json",
                    str(gate_overlay_out / "evidence_qwen3vl_guarded_overlay_summary.json"),
                ],
                env=base_env,
            )
        if selected == "export":
            return _run_command(
                [
                    py_helper,
                    "-m",
                    "wrbench.eval.scoring.export_runtime_v2_evidence_first",
                    "--scores-v7",
                    str(qwen35_out / "scores_v7_candidate_runtime_v2_probe_score_available.json"),
                    "--evidence-jsonl",
                    str(gate_overlay_out / "evidence_qwen3vl_guarded_overlay.jsonl"),
                    "--manifest-path",
                    str(manifest),
                    "--out-dir",
                    str(final_out),
                ],
                env=base_env,
            )
        _err(f"[error] unknown stage: {selected}")
        return 2

    stages_to_run = ALL_STAGE_ORDER if stage == "all" else (stage,)
    for selected_stage in stages_to_run:
        stage_rc = _run_stage(selected_stage)
        if stage_rc != 0:
            return stage_rc

    print(f"[metric] out_dir={out_dir}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _err("[error] stage argument is required")
        _err(USAGE)
        return 2
    stage = args[0]
    if stage in {"-h", "--help", "help"}:
        print(USAGE, flush=True)
        return 0
    return run_d3d6_pipeline(stage=stage, env=os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
