#!/usr/bin/env python
"""Run Natural-25 camera benchmark generation for one model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def _project_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


if str(_project_src()) not in sys.path:
    sys.path.insert(0, str(_project_src()))

import wrbench  # noqa: E402
from wrbench.benchmark import (  # noqa: E402
    NATURAL25_CAMERA_COMBOS,
    Natural25CameraTask,
    load_natural25_camera_scope,
    natural25_camera_tasks,
    natural25_camera_tasks_from_scope,
)
from wrbench.datasets import NATURAL25_PROMPT_PROFILES, PROMPT_PROFILE_T2V_LAYOUT_ANCHOR  # noqa: E402
from wrbench.datasets import natural25_first_frame_path  # noqa: E402
from wrbench.registry import input_kind  # noqa: E402
from wrbench.source_video_tasks import (  # noqa: E402
    SourceVideoBinding,
    SourceVideoTaskMap,
    load_source_video_task_map,
)


def _jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _task_output_path(out_dir: Path, model: str, task: Natural25CameraTask) -> Path:
    return out_dir / model / "videos" / f"{task.output_id}.mp4"


def _model_input_label(kind: str) -> str:
    if kind == "none":
        return "T2V"
    if kind == "image":
        return "TI2V"
    if kind == "source_video":
        return "TV2V"
    raise ValueError(f"unsupported input_kind {kind!r}")


def _paper_scope_model_id(model: str) -> str:
    """Map public registry keys to frozen paper-scope model identifiers."""
    if model == "inspatio-world":
        return "inspatio_world_14b"
    return model.replace("-", "_")


def _write_source_provenance_to_camera_sidecar(
    artifacts: dict[str, Any],
    binding: SourceVideoBinding | None,
) -> None:
    if binding is None:
        return
    sidecar_path = artifacts.get("camera_sidecar_path")
    if not sidecar_path:
        raise ValueError("source-video generation did not produce camera_sidecar_path")
    path = Path(str(sidecar_path))
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        raise ValueError(f"camera sidecar must be a JSON object: {path}")
    sidecar["source_video_provenance"] = binding.sidecar_provenance()
    path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _control_condition_type(payload_summary: dict[str, Any]) -> str | None:
    official_kind = str(payload_summary.get("official_input_kind") or "").strip()
    if not official_kind:
        return None
    return official_kind.replace("_", "-")


def _update_row_from_camera_sidecar(row: dict[str, Any], artifacts: dict[str, Any]) -> None:
    sidecar_path = artifacts.get("camera_sidecar_path")
    if not sidecar_path:
        return
    path = Path(str(sidecar_path))
    if not path.is_file():
        return
    sidecar = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, dict):
        return
    row["camera_sidecar_path"] = str(path)
    for field_name in (
        "target_pose_path",
        "trajectory_c2w_path",
        "camera_trajectory_path",
        "target_coordinate_convention",
        "coordinate_convention",
        "target_certification_status",
        "target_role",
        "evidence_level",
        "control_family",
        "control_direction",
        "control_profile",
        "yaw_peak_deg",
        "target_yaw_peak_deg",
        "num_frames",
        "fps",
        "image_size",
        "fov",
        "trajectory_sampling_rule",
        "adapter_provenance",
    ):
        if field_name in sidecar:
            row[field_name] = sidecar[field_name]


def _compile_kwargs(
    model: str,
    task: Natural25CameraTask,
    out_path: Path,
    *,
    dry_run: bool,
    runtime_config: Path | None,
    source_video_task_map: SourceVideoTaskMap | None,
) -> tuple[dict, SourceVideoBinding | None]:
    record = wrbench.model_record(model)
    camera_kwargs = {"frames": int(record.default_frames)}
    if task.camera_type in {"yaw_LR", "yaw_RL"} and task.stress_yaw_deg is not None:
        camera_kwargs["peak_deg"] = float(task.stress_yaw_deg)
    camera = wrbench.presets.build_preset(task.preset, **camera_kwargs)
    work_dir = out_path.with_suffix(out_path.suffix + ".wrbench_work")
    kwargs = {
        "model": model,
        "camera": camera,
        "camera_type": task.camera_type,
        "out": out_path,
        "prompt": task.prompt,
        "work_dir": work_dir,
        "runtime_config": runtime_config,
        "dry_run": dry_run,
    }
    kind = input_kind(model)
    binding: SourceVideoBinding | None = None
    if kind == "image":
        kwargs["image"] = str(natural25_first_frame_path(task.family_id))
    elif kind == "source_video":
        if source_video_task_map is None:
            raise ValueError(
                f"{model} is source-video input; provide both --source-video-task-map and "
                "--source-video-root. Repeated first-frame clips are not valid benchmark TV2V inputs."
            )
        binding = source_video_task_map.resolve(
            model=model,
            task_variant_id=task.variant_id,
            source_video_usage=record.source_video_usage,
        )
        kwargs.update(binding.compile_kwargs())
    elif kind != "none":
        raise ValueError(f"{model} has unsupported input_kind {kind!r}")
    return kwargs, binding


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="WRBench model key or alias.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Run output root.")
    parser.add_argument("--runtime-config", type=Path, help="Explicit wrbench.runtime.json path for real generation.")
    parser.add_argument(
        "--source-video-task-map",
        type=Path,
        help="Explicit task_variant_id-to-source map for source-video models.",
    )
    parser.add_argument(
        "--source-video-root",
        type=Path,
        help="Explicit root for relative source paths in --source-video-task-map.",
    )
    parser.add_argument("--variants", type=Path, help="Override Natural-25 variants JSONL.")
    parser.add_argument("--prompt-profile", required=True, choices=NATURAL25_PROMPT_PROFILES)
    scope_group = parser.add_mutually_exclusive_group(required=True)
    scope_group.add_argument("--camera-scope", type=Path, help="Explicit Natural-25 camera scope JSON.")
    scope_group.add_argument("--cameras", nargs="+", choices=NATURAL25_CAMERA_COMBOS)
    execution_group = parser.add_mutually_exclusive_group(required=True)
    execution_group.add_argument("--dry-run", action="store_true", help="Compile sidecars and payloads without generation.")
    execution_group.add_argument("--no-dry-run", action="store_true", help="Run the configured generation backend.")
    existing_group = parser.add_mutually_exclusive_group(required=True)
    existing_group.add_argument("--skip-existing", action="store_true", help="Skip existing non-empty MP4 outputs.")
    existing_group.add_argument("--overwrite-existing", action="store_true", help="Regenerate existing outputs.")
    failure_group = parser.add_mutually_exclusive_group(required=True)
    failure_group.add_argument("--continue-on-error", action="store_true", help="Record failures and continue remaining tasks.")
    failure_group.add_argument("--fail-fast", action="store_true", help="Stop after the first failed task.")
    parser.add_argument("--limit", type=int, help="Maximum tasks to process after sharding.")
    parser.add_argument("--shard-index", type=int, required=True, help="Zero-based shard index.")
    parser.add_argument("--num-shards", type=int, required=True, help="Total number of shards.")
    args = parser.parse_args(argv)

    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    if args.no_dry_run and args.runtime_config is None:
        raise ValueError("--runtime-config is required with --no-dry-run")
    if (args.source_video_task_map is None) != (args.source_video_root is None):
        raise ValueError("--source-video-task-map and --source-video-root must be supplied together")

    model = wrbench.canonical_model_key(args.model)
    kind = input_kind(model)
    record = wrbench.model_record(model)
    source_video_task_map = (
        load_source_video_task_map(
            args.source_video_task_map,
            source_root=args.source_video_root,
        )
        if args.source_video_task_map is not None and args.source_video_root is not None
        else None
    )
    if source_video_task_map is not None and kind != "source_video":
        raise ValueError(f"{model} does not use source-video input; remove the source-video task map/root")
    if args.prompt_profile == PROMPT_PROFILE_T2V_LAYOUT_ANCHOR and kind != "none":
        raise ValueError("--prompt-profile t2v_layout_anchor is only valid for T2V prompt-only models")
    if args.camera_scope is not None:
        camera_scope = load_natural25_camera_scope(args.camera_scope)
        if (
            camera_scope.applicable_models is not None
            and _paper_scope_model_id(model) not in camera_scope.applicable_models
        ):
            raise ValueError(
                f"{model} is not included in camera scope {camera_scope.scope_id!r}; "
                f"applicable models: {', '.join(camera_scope.applicable_models)}"
            )
        camera_scope_id = camera_scope.scope_id
        camera_scope_path = str(args.camera_scope)
        tasks = natural25_camera_tasks_from_scope(
            camera_scope=camera_scope,
            variants_path=args.variants,
            prompt_profile=args.prompt_profile,
        )
    else:
        camera_scope_id = "explicit_camera_list"
        camera_scope_path = ""
        tasks = natural25_camera_tasks(
            variants_path=args.variants,
            cameras=args.cameras,
            prompt_profile=args.prompt_profile,
        )
    total_tasks = len(tasks)
    tasks = [task for idx, task in enumerate(tasks) if idx % args.num_shards == args.shard_index]
    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]

    run_dir = args.out_dir / model
    manifest_path = run_dir / f"manifest.shard{args.shard_index:02d}.jsonl"
    summary_path = run_dir / f"summary.shard{args.shard_index:02d}.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_input = record.model_input or _model_input_label(kind)

    started = time.time()
    counts = {"ok": 0, "failed": 0, "skipped": 0}
    print(
        json.dumps(
            {
                "model": model,
                "input_kind": kind,
                "model_input": model_input,
                "source_video_usage": record.source_video_usage,
                "dry_run": args.dry_run,
                "camera_scope_id": camera_scope_id,
                "camera_scope_path": camera_scope_path,
                "prompt_profile": args.prompt_profile,
                "total_unsharded_tasks": total_tasks,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "shard_tasks": len(tasks),
                "manifest": str(manifest_path),
                "source_video_task_map_path": str(args.source_video_task_map) if args.source_video_task_map else "",
                "source_video_root": str(args.source_video_root) if args.source_video_root else "",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    for ordinal, task in enumerate(tasks, start=1):
        out_path = _task_output_path(args.out_dir, model, task)
        row = {
            "schema_version": 1,
            "model": model,
            "display_name": record.key,
            "video_id": task.output_id,
            "variant_id": task.variant_id,
            "family_id": task.family_id,
            "reasoning_tier": task.reasoning_tier,
            "event_delta": task.event_delta,
            "divergence_id": task.divergence_id,
            "oov_gap": task.oov_gap,
            "camera": task.camera,
            "camera_type": task.camera_type,
            "camera_preset": task.preset,
            "stress_axis": task.stress_axis,
            "stress_yaw_deg": task.stress_yaw_deg,
            "camera_scope_id": task.camera_scope_id,
            "path": str(out_path),
            "world_state_prompt": task.world_state_prompt,
            "expected_state": task.expected_state,
            "prompt_profile_id": task.prompt_profile_id,
            "ti2v_prompt": task.ti2v_prompt,
            "prompt": task.prompt,
            "input_kind": kind,
            "model_input": model_input,
            "source_video_usage": record.source_video_usage,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "ordinal": ordinal,
            "status": "started",
        }
        try:
            if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                row["status"] = "skipped_existing"
                counts["skipped"] += 1
            else:
                compile_kwargs, source_binding = _compile_kwargs(
                    model,
                    task,
                    out_path,
                    dry_run=args.dry_run,
                    runtime_config=args.runtime_config,
                    source_video_task_map=source_video_task_map,
                )
                if source_binding is not None:
                    row.update(source_binding.manifest_metadata())
                result = wrbench.compile_camera(**compile_kwargs)
                payload = result["payload"]
                row["status"] = "ok"
                row["dry_run"] = bool(result["dry_run"])
                row["payload_type"] = payload.payload_type
                row["official_camera_entrypoint"] = payload.official_camera_entrypoint
                row["model_payload_summary"] = dict(payload.metadata["model_payload_summary"])
                row["model_control_timeline"] = dict(payload.metadata["model_control_timeline"])
                row["artifacts"] = dict(result.get("artifacts") or {})
                row["generation"] = result.get("generation")
                _write_source_provenance_to_camera_sidecar(row["artifacts"], source_binding)
                condition_type = _control_condition_type(row["model_payload_summary"])
                if condition_type:
                    row["control_condition_type"] = condition_type
                _update_row_from_camera_sidecar(row, row["artifacts"])
                counts["ok"] += 1
        except Exception as exc:  # noqa: BLE001 - batch manifest should record and continue.
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            counts["failed"] += 1
        _jsonl_append(manifest_path, row)
        print(
            json.dumps(
                {
                    "ordinal": ordinal,
                    "tasks": len(tasks),
                    "output_id": task.output_id,
                    "camera": task.camera,
                    "camera_type": task.camera_type,
                    "stress_yaw_deg": task.stress_yaw_deg,
                    "status": row["status"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if row["status"] == "failed" and args.fail_fast:
            break

    summary = {
        "schema_version": 1,
        "model": model,
        "input_kind": kind,
        "model_input": model_input,
        "source_video_usage": record.source_video_usage,
        "dry_run": args.dry_run,
        "failure_policy": "fail_fast" if args.fail_fast else "continue_on_error",
        "existing_output_policy": "skip_existing" if args.skip_existing else "overwrite_existing",
        "camera_scope_id": camera_scope_id,
        "camera_scope_path": camera_scope_path,
        "prompt_profile": args.prompt_profile,
        "total_unsharded_tasks": total_tasks,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "shard_tasks": len(tasks),
        "counts": counts,
        "manifest": str(manifest_path),
        "elapsed_seconds": time.time() - started,
        "runtime_config_path": str(args.runtime_config) if args.runtime_config else "",
        "source_video_task_map_path": str(args.source_video_task_map) if args.source_video_task_map else "",
        "source_video_root": str(args.source_video_root) if args.source_video_root else "",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, sort_keys=True), flush=True)
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
