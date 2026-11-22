"""CLI subcommand: eval (D1-D6 evaluation, submission validation, plugins, contract)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from wrbench._cli.common import collect_plugins_summary, print_plugins_summary


def _cmd_eval(args: argparse.Namespace) -> int:
    from wrbench.eval.runtime import (
        build_table,
        contract_path,
        d1_camalign_score,
        d1_score,
        d1_vggt_batch,
        d2_extract,
        d3d6_score,
        eval_run,
        require_eval_runtime,
    )

    runtime_path = Path(args.runtime_config) if args.runtime_config else None

    if args.eval_command == "contract":
        path = contract_path()
        if args.json:
            print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2))
        else:
            md = path.with_suffix(".md")
            print(md.read_text(encoding="utf-8") if md.is_file() else path.read_text(encoding="utf-8"))
        return 0

    if args.eval_command == "plugins":
        return print_plugins_summary(collect_plugins_summary(), as_json=bool(args.json))

    if args.eval_command == "validate":
        from wrbench.tasks_submission import validate_video_submission

        report = validate_video_submission(
            args.manifest,
            video_root=args.video_root,
            check_files=not args.no_check_files,
            check_sha256=bool(args.check_sha256),
        )
        if args.json:
            print(json.dumps(report, indent=2))
        else:
            status = "PASS" if report["ok"] else "FAIL"
            models_str = ", ".join(report.get("models", [])) or "none"
            print(f"status   : {status}")
            print(f"rows     : {report.get('row_count', 0)}")
            print(f"models   : {models_str}")
            for warning in report.get("warnings", []):
                print(f"warning  : {warning}")
            for err in report.get("errors", []):
                print(f"error    : {err}")
        return 0 if report.get("ok") else 1

    if args.eval_command == "table":
        if not args.runtime_scores or not args.out_csv or not args.out_md or not args.out_summary:
            print(
                "eval table requires --d3d6-scores (or --vlm-scores / --runtime-scores) "
                "--out-csv --out-md --out-summary",
                file=sys.stderr,
            )
            return 2
        argv = [
            "--d3d6-scores",
            str(args.runtime_scores),
            "--out-csv",
            str(args.out_csv),
            "--out-md",
            str(args.out_md),
            "--out-summary",
            str(args.out_summary),
        ]
        if args.d1_scores:
            argv.extend(["--d1-scores", str(args.d1_scores)])
        if args.d1_camalign_scores:
            argv.extend(["--d1-camalign-scores", str(args.d1_camalign_scores)])
        if args.d2_scores:
            argv.extend(["--d2-scores", str(args.d2_scores)])
        return build_table(argv)

    if args.eval_command == "d1":
        return d1_score(
            input_jsonl=Path(args.input_jsonl),
            output_jsonl=Path(args.output_jsonl),
            summary_csv=Path(args.summary_csv),
            pose_cache_root=Path(args.pose_cache_root),
            pose_backend=args.pose_backend,
            poses_file=args.poses_file,
            default_frames=args.default_frames,
            sidecar_profile_gate=args.sidecar_profile_gate,
            target_fallback_mode=args.target_fallback_mode,
            predicted_pose_type=args.predicted_pose_type,
            predicted_camera_convention=args.predicted_camera_convention,
            target_camera_convention=args.target_camera_convention,
            rot_scale_deg=args.rot_scale_deg,
            trans_scale=args.trans_scale,
            yaw_weak_threshold_deg=args.yaw_weak_threshold_deg,
            pan_weak_threshold=args.pan_weak_threshold,
            static_rot_threshold_deg=args.static_rot_threshold_deg,
            static_trans_threshold=args.static_trans_threshold,
        )

    if args.eval_command == "d1-camalign":
        return d1_camalign_score(
            input_jsonl=Path(args.input_jsonl),
            output_jsonl=Path(args.output_jsonl),
            pose_cache_root=Path(args.pose_cache_root),
            poses_file=args.poses_file,
            scorer_version=args.scorer_version,
            summary_json=Path(args.summary_json) if args.summary_json else None,
        )

    eval_runtime = require_eval_runtime(runtime_path)

    if args.eval_command in {"pose", "d1-vggt"}:
        return d1_vggt_batch(
            eval_runtime=eval_runtime,
            input_jsonl=Path(args.input_jsonl),
            output_root=Path(args.output_root),
            cache_root=Path(args.cache_root),
            execution_mode=args.execution_mode,
            video_root=Path(args.video_root) if args.video_root else None,
            max_frames=args.max_frames,
        )

    if args.eval_command == "d2":
        return d2_extract(
            eval_runtime=eval_runtime,
            videos_manifest=Path(args.videos_manifest),
            out_jsonl=Path(args.out_jsonl),
            model_dir=Path(args.model_dir) if args.model_dir else None,
            video_root=Path(args.video_root) if args.video_root else None,
        )

    if args.eval_command == "d3d6":
        return d3d6_score(
            eval_runtime=eval_runtime,
            manifest=Path(args.manifest),
            out_dir=Path(args.out_dir),
            stage=args.stage,
            scorer_profile=args.scorer_profile,
        )

    if args.eval_command == "run":
        return eval_run(
            eval_runtime=eval_runtime,
            manifest=Path(args.manifest),
            out_dir=Path(args.out_dir),
            scorer_profile=args.scorer_profile,
            sidecar_profile_gate=args.sidecar_profile_gate,
            target_fallback_mode=args.target_fallback_mode,
            video_root=Path(args.video_root) if args.video_root else None,
            camalign_scorer_version=args.camalign_scorer_version,
        )

    print(f"eval: unknown sub-command {args.eval_command!r}", file=sys.stderr)
    return 2


def register_eval_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the eval subparser and all its sub-commands."""
    p_eval = sub.add_parser(
        "eval",
        help="Run WRBench D1-D6 diagnostic evaluation and table aggregation.",
    )
    eval_sub = p_eval.add_subparsers(dest="eval_command", metavar="<eval-command>")
    eval_sub.required = True
    p_eval.add_argument(
        "--runtime-config",
        dest="runtime_config",
        help="Explicit path to wrbench.runtime.json for eval scorer runtime paths.",
    )

    from wrbench.eval.plugins import D1_TARGET_MODE_HELP, D1_TARGET_MODES, DEFAULT_D1_SCORER_CONFIG

    p_eval_contract = eval_sub.add_parser("contract", help="Print the D1-D6 metric contract.")
    p_eval_contract.add_argument("--json", action="store_true", help="Print JSON contract.")

    p_eval_plugins = eval_sub.add_parser(
        "plugins", help="List registered generation and evaluation plugins."
    )
    p_eval_plugins.add_argument("--json", action="store_true", help="Print JSON plugin summary.")

    p_eval_validate = eval_sub.add_parser(
        "validate",
        help="Validate a video submission manifest (JSON/JSONL) before running D1-D6 evaluation.",
    )
    p_eval_validate.add_argument("--manifest", required=True, metavar="PATH", help="Manifest JSON/JSONL path.")
    p_eval_validate.add_argument("--video-root", dest="video_root", metavar="DIR", help="Base directory for relative video paths.")
    p_eval_validate.add_argument(
        "--no-check-files",
        action="store_true",
        dest="no_check_files",
        help="Validate manifest schema only without checking that video files exist on disk.",
    )
    p_eval_validate.add_argument(
        "--check-sha256",
        action="store_true",
        dest="check_sha256",
        help="Verify SHA-256 hashes for rows that declare a sha256 field.",
    )
    p_eval_validate.add_argument("--json", action="store_true", help="Print validation report as JSON.")

    p_eval_d1 = eval_sub.add_parser(
        "d1", help="Score D1 camera accuracy from cached poses and camera targets."
    )
    p_eval_d1.add_argument("--input-jsonl", required=True)
    p_eval_d1.add_argument("--output-jsonl", required=True)
    p_eval_d1.add_argument("--summary-csv", required=True)
    p_eval_d1.add_argument("--pose-cache-root", required=True)
    p_eval_d1.add_argument("--pose-backend", default=DEFAULT_D1_SCORER_CONFIG["pose_backend"])
    p_eval_d1.add_argument("--poses-file", default=DEFAULT_D1_SCORER_CONFIG["poses_file"])
    p_eval_d1.add_argument("--default-frames", type=int, default=DEFAULT_D1_SCORER_CONFIG["default_frames"])
    p_eval_d1.add_argument("--predicted-pose-type", default=DEFAULT_D1_SCORER_CONFIG["predicted_pose_type"])
    p_eval_d1.add_argument("--predicted-camera-convention", default=DEFAULT_D1_SCORER_CONFIG["predicted_camera_convention"])
    p_eval_d1.add_argument("--target-camera-convention", default=DEFAULT_D1_SCORER_CONFIG["target_camera_convention"])
    p_eval_d1.add_argument("--rot-scale-deg", type=float, default=DEFAULT_D1_SCORER_CONFIG["rot_scale_deg"])
    p_eval_d1.add_argument("--trans-scale", type=float, default=DEFAULT_D1_SCORER_CONFIG["trans_scale"])
    p_eval_d1.add_argument("--yaw-weak-threshold-deg", type=float, default=DEFAULT_D1_SCORER_CONFIG["yaw_weak_threshold_deg"])
    p_eval_d1.add_argument("--pan-weak-threshold", type=float, default=DEFAULT_D1_SCORER_CONFIG["pan_weak_threshold"])
    p_eval_d1.add_argument("--static-rot-threshold-deg", type=float, default=DEFAULT_D1_SCORER_CONFIG["static_rot_threshold_deg"])
    p_eval_d1.add_argument("--static-trans-threshold", type=float, default=DEFAULT_D1_SCORER_CONFIG["static_trans_threshold"])
    p_eval_d1.add_argument(
        "--sidecar-profile-gate",
        choices=("main", "certified_opencv"),
        default="main",
        help=(
            "Target sidecar validation gate. 'main' is the canonical main-table profile; "
            "'certified_opencv' accepts certified OpenCV C2W sidecars after external QC."
        ),
    )
    p_eval_d1.add_argument(
        "--target-mode",
        "--target-fallback-mode",
        dest="target_fallback_mode",
        choices=D1_TARGET_MODES,
        default=DEFAULT_D1_SCORER_CONFIG["target_fallback_mode"],
        help=D1_TARGET_MODE_HELP,
    )

    p_eval_camalign = eval_sub.add_parser(
        "d1-camalign",
        help="Score D1 prompt-camera alignment (CamAlign) from pose cache.",
    )
    p_eval_camalign.add_argument("--input-jsonl", required=True)
    p_eval_camalign.add_argument("--output-jsonl", required=True)
    p_eval_camalign.add_argument("--pose-cache-root", required=True)
    p_eval_camalign.add_argument("--poses-file", default="poses.npy")
    p_eval_camalign.add_argument("--scorer-version", choices=("v2_lite", "v1"), default="v2_lite")
    p_eval_camalign.add_argument("--summary-json")

    p_eval_vggt = eval_sub.add_parser("pose", aliases=["d1-vggt"], help="Export camera poses for D1 scoring.")
    p_eval_vggt.add_argument("--input-jsonl", required=True)
    p_eval_vggt.add_argument("--output-root", required=True)
    p_eval_vggt.add_argument("--cache-root", required=True)
    p_eval_vggt.add_argument("--video-root")
    p_eval_vggt.add_argument(
        "--execution-mode",
        choices=("subprocess", "inprocess"),
    )
    p_eval_vggt.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame cap per video (defaults to 49; set 0 for unconstrained full-frame extraction).",
    )

    p_eval_d2 = eval_sub.add_parser("d2", help="Score D2 visual integrity.")
    p_eval_d2.add_argument("--videos-manifest", required=True)
    p_eval_d2.add_argument("--out-jsonl", required=True)
    p_eval_d2.add_argument("--model-dir")
    p_eval_d2.add_argument("--video-root")

    p_eval_run = eval_sub.add_parser(
        "run",
        help="Run full eval pipeline: D1 pose -> D1 score -> D2 -> D3-D6 -> table.",
    )
    p_eval_run.add_argument("--manifest", required=True, help="JSON manifest of videos to score.")
    p_eval_run.add_argument("--out-dir", required=True, help="Output directory for all eval artifacts.")
    p_eval_run.add_argument(
        "--scorer-profile",
        default=None,
        choices=("default", "wrbench_default", "ablation_manifest_metadata", "custom"),
        help=(
            "Evaluation profile. Defaults to 'wrbench_default' (or 'custom' automatically "
            "when eval.backends overrides a stage); 'default' aliases 'wrbench_default'."
        ),
    )
    p_eval_run.add_argument(
        "--sidecar-profile-gate",
        choices=("main", "certified_opencv"),
        default="main",
        help=(
            "Target sidecar validation gate. 'main' is the canonical main-table profile; "
            "'certified_opencv' accepts certified OpenCV C2W sidecars after external QC."
        ),
    )
    p_eval_run.add_argument(
        "--target-mode",
        "--target-fallback-mode",
        dest="target_fallback_mode",
        choices=D1_TARGET_MODES,
        default=DEFAULT_D1_SCORER_CONFIG["target_fallback_mode"],
        help=D1_TARGET_MODE_HELP,
    )
    p_eval_run.add_argument("--video-root", help="Optional base directory for relative manifest video paths.")
    p_eval_run.add_argument("--camalign-scorer-version", choices=("v2_lite", "v1"), default="v2_lite")

    p_eval_d3d6 = eval_sub.add_parser("d3d6", help="Run individual D3-D6 VLM scoring stages.")
    p_eval_d3d6.add_argument("--manifest", required=True)
    p_eval_d3d6.add_argument("--out-dir", required=True)
    p_eval_d3d6.add_argument(
        "--stage",
        default="all",
        choices=(
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
        ),
    )
    p_eval_d3d6.add_argument(
        "--scorer-profile",
        default="wrbench_default",
        choices=("default", "wrbench_default", "ablation_manifest_metadata", "custom"),
    )

    p_eval_table = eval_sub.add_parser("table", help="Build D1-D6 main benchmark table.")
    p_eval_table.add_argument(
        "--d3d6-scores",
        "--vlm-scores",
        "--runtime-scores",
        dest="runtime_scores",
        required=True,
        help="D3-D6 VLM score export JSON.",
    )
    p_eval_table.add_argument("--d1-scores")
    p_eval_table.add_argument("--d1-camalign-scores")
    p_eval_table.add_argument("--d2-scores")
    p_eval_table.add_argument("--out-csv", required=True)
    p_eval_table.add_argument("--out-md", required=True)
    p_eval_table.add_argument("--out-summary", required=True)
