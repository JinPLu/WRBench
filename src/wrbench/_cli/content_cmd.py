"""CLI subcommands: prompt, firstframe, profile, and profile-summary."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _cmd_profile(args: argparse.Namespace) -> int:
    from wrbench.profiling import run_profiled_command

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        print("profile: missing command after --", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or "run"
    summary = out_dir / f"{stem}.resource_profile.json"
    trace = out_dir / f"{stem}.resource_trace.jsonl" if not args.no_trace else None
    events = out_dir / f"{stem}.stage_events.jsonl"
    log = out_dir / f"{stem}.generation.log" if args.log else None

    profile = run_profiled_command(
        cmd,
        cwd=args.cwd,
        summary_path=summary,
        trace_path=trace,
        stage_events_path=events,
        log_path=log,
        run_identity={
            "model": args.model or "",
            "profile": args.profile or "",
            "camera": args.camera or "",
            "scene_id": args.scene_id or "",
            "gpu_width": args.gpu_width,
            "output_video_seconds": args.output_video_seconds,
        },
        sampling_interval_seconds=args.sampling_interval,
        generation_status=args.generation_status,
    )
    if args.json:
        print(json.dumps(profile, indent=2))
    else:
        dm = profile.get("derived_metrics", {})
        print(f"Summary: {summary}")
        print(f"exit_code={profile['status']['exit_code']} wall={profile['status']['wall_time_seconds']:.3f}s")
        bg = dm.get("benchmark_generation_seconds")
        if bg is not None:
            print(f"benchmark_generation_seconds={bg:.3f}")
        gpus = dm.get("benchmark_gpu_seconds_per_output_second")
        if gpus is not None:
            print(f"benchmark_gpu_seconds_per_output_second={gpus:.3f}")
    return int(profile["status"]["exit_code"])


def _cmd_profile_summary(args: argparse.Namespace) -> int:
    from wrbench.profiling import format_rows, load_profiles, summarize

    rows = summarize(load_profiles([Path(p) for p in args.paths]))
    text = format_rows(rows, args.format)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


def _cmd_prompt(args: argparse.Namespace) -> int:
    if args.prompt_command == "camera":
        from wrbench.prompts.camera_text import (
            assemble_ti2v_prompt,
            build_prompt_to_send,
            preset_camera_text,
        )

        if args.preset:
            if not args.pronoun or not args.offscreen_area:
                print("prompt camera --preset requires --pronoun and --offscreen-area", file=sys.stderr)
                return 2
            text = preset_camera_text(args.preset, pronoun=args.pronoun, offscreen_area=args.offscreen_area)
            result = {"preset": args.preset, "camera_text": text}
        elif args.assemble:
            if not all((args.scene_start, args.event, args.pronoun, args.offscreen_area, args.gap)):
                print(
                    "prompt camera --assemble requires --scene-start, --event, --pronoun, --offscreen-area, and --gap",
                    file=sys.stderr,
                )
                return 2
            result = {
                "ti2v_prompt": assemble_ti2v_prompt(
                    args.scene_start,
                    args.event,
                    args.pronoun,
                    args.offscreen_area,
                    args.gap,
                )
            }
        elif args.model and args.source_prompt:
            if not args.camera_motion:
                print("prompt camera --model + --source-prompt requires --camera-motion", file=sys.stderr)
                return 2
            result = {
                "prompt_to_send": build_prompt_to_send(
                    args.source_prompt,
                    args.camera_motion,
                    model=args.model,
                )
            }
        else:
            print("prompt camera: use --preset, --assemble, or --model + --source-prompt", file=sys.stderr)
            return 2
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.prompt_command == "scene":
        from wrbench.prompts.scene import enrich_family_with_t2i_scene

        family = json.loads(Path(args.family_json).read_text(encoding="utf-8"))
        enriched = enrich_family_with_t2i_scene(
            family,
            model=args.model,
            temperature=args.temperature,
            provider=args.provider,
            api_key=args.api_key,
            base_url=args.base_url,
        )
        print(json.dumps(enriched, indent=2, ensure_ascii=False))
        return 0

    if args.prompt_command == "task":
        from wrbench.prompts.task import (
            generate_variants_deterministic,
            load_jsonl,
            write_jsonl,
        )

        if args.deterministic:
            from wrbench.datasets import build_natural25_candidates, natural25_families_path

            candidates_path = args.candidates_json
            families_path = args.families_jsonl
            if bool(candidates_path) ^ bool(families_path):
                print(
                    "prompt task --deterministic: provide both --candidates-json and "
                    "--families-jsonl, or omit both to use bundled Natural-25 data.",
                    file=sys.stderr,
                )
                return 2
            if candidates_path and families_path:
                candidates = json.loads(Path(candidates_path).read_text(encoding="utf-8"))
                families = {r["family_id"]: r for r in load_jsonl(families_path)}
            else:
                candidates = build_natural25_candidates()
                families = {r["family_id"]: r for r in load_jsonl(natural25_families_path())}
            missing = [c["candidate_id"] for c in candidates if c["candidate_id"] not in families]
            if missing:
                print(
                    f"prompt task --deterministic: warning: {len(missing)} candidate(s) "
                    f"have no matching family and will be skipped.",
                    file=sys.stderr,
                )
            variants = generate_variants_deterministic(candidates, families)
            if args.output:
                write_jsonl(args.output, variants)
                print(f"Wrote {len(variants)} variants to {args.output}")
            else:
                print(json.dumps(variants[:3], indent=2, ensure_ascii=False))
                if len(variants) > 3:
                    print(f"... and {len(variants) - 3} more")
            return 0

        print("prompt task: only --deterministic path is implemented without LLM; use scene+LLM for tier groups", file=sys.stderr)
        return 2

    print(f"Unknown prompt subcommand: {args.prompt_command}", file=sys.stderr)
    return 2


def _cmd_firstframe(args: argparse.Namespace) -> int:
    from wrbench.firstframe import (
        generate_first_frame,
        generate_first_frames_from_families,
        write_manifest,
    )
    from wrbench.prompts.task import load_jsonl

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.families_jsonl:
        families = list(load_jsonl(args.families_jsonl))
        manifests = generate_first_frames_from_families(
            families,
            out_dir=out_dir,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            endpoint=args.endpoint,
            size=args.size,
            n=args.n,
            skip_existing=args.skip_existing,
        )
    elif args.prompt and args.family_id:
        manifests = [
            generate_first_frame(
                family_id=args.family_id,
                prompt=args.prompt,
                out_dir=out_dir,
                provider=args.provider,
                model=args.model,
                api_key=args.api_key,
                endpoint=args.endpoint,
                size=args.size,
                n=args.n,
            )
        ]
    else:
        print("firstframe: provide --families-jsonl or --family-id + --prompt", file=sys.stderr)
        return 2

    manifest_path = out_dir / "first_frames_manifest.json"
    write_manifest(manifest_path, manifests)
    print(f"Generated {len(manifests)} frame(s); manifest: {manifest_path}")
    for m in manifests:
        print(f"  {m.family_id} -> {m.image_path}")
    return 0


def register_content_parsers(sub: argparse._SubParsersAction[Any]) -> None:
    """Register profile, profile-summary, prompt, and firstframe subparsers."""
    p_prof = sub.add_parser("profile", help="Profile a generation command.")
    p_prof.add_argument("--out-dir", required=True, metavar="DIR", help="Output directory for profile artifacts.")
    p_prof.add_argument("--name", required=True, metavar="NAME", help="Artifact filename stem.")
    p_prof.add_argument("--model", metavar="MODEL")
    p_prof.add_argument("--profile", metavar="PROFILE")
    p_prof.add_argument("--camera", metavar="CAMERA")
    p_prof.add_argument("--scene-id", dest="scene_id", metavar="ID")
    p_prof.add_argument("--gpu-width", type=int, required=True, dest="gpu_width")
    p_prof.add_argument("--output-video-seconds", type=float, dest="output_video_seconds")
    p_prof.add_argument("--generation-status", dest="generation_status")
    p_prof.add_argument("--sampling-interval", type=float, required=True, dest="sampling_interval")
    p_prof.add_argument("--cwd", required=True, metavar="DIR")
    p_prof.add_argument("--log", action="store_true", help="Capture command stdout/stderr to a log file.")
    p_prof.add_argument("--no-trace", action="store_true", help="Skip GPU trace JSONL.")
    p_prof.add_argument("--json", action="store_true", help="Print full profile JSON to stdout.")
    p_prof.add_argument("cmd", nargs=argparse.REMAINDER, help="Command after --")

    p_psum = sub.add_parser("profile-summary", help="Summarize resource profile JSON files.")
    p_psum.add_argument("paths", nargs="+", metavar="PATH", help="Profile JSON, JSONL, or directory.")
    p_psum.add_argument("--format", choices=("json", "csv", "markdown"), required=True)
    p_psum.add_argument("--output", metavar="PATH")

    p_prompt = sub.add_parser("prompt", help="Generate scene/task/camera prompts.")
    p_prompt_sub = p_prompt.add_subparsers(dest="prompt_command", metavar="<kind>")
    p_prompt_sub.required = True

    p_cam = p_prompt_sub.add_parser("camera", help="Camera prompt text or API assembly.")
    p_cam.add_argument("--preset", metavar="NAME", help="wrbench preset name (yaw_LR, static, …).")
    p_cam.add_argument("--pronoun")
    p_cam.add_argument("--offscreen-area", dest="offscreen_area")
    p_cam.add_argument("--assemble", action="store_true", help="Assemble ti2v_prompt from parts.")
    p_cam.add_argument("--scene-start", dest="scene_start")
    p_cam.add_argument("--event")
    p_cam.add_argument("--gap")
    p_cam.add_argument("--model", metavar="API_MODEL", help="API model for prompt_to_send assembly.")
    p_cam.add_argument("--source-prompt", dest="source_prompt", metavar="TEXT")
    p_cam.add_argument("--camera-motion", dest="camera_motion")

    p_scene = p_prompt_sub.add_parser("scene", help="Generate t2i_scene via LLM (requires wrbench[prompts]).")
    p_scene.add_argument("--family-json", required=True, dest="family_json", metavar="PATH")
    p_scene.add_argument("--model", required=True)
    p_scene.add_argument("--provider", required=True)
    p_scene.add_argument("--base-url", required=True, dest="base_url")
    p_scene.add_argument("--api-key", required=True, dest="api_key")
    p_scene.add_argument("--temperature", type=float, required=True)

    p_task = p_prompt_sub.add_parser("task", help="Generate ti2v variant prompts.")
    p_task.add_argument("--deterministic", action="store_true", help="Deterministic Natural-25 style path.")
    p_task.add_argument(
        "--candidates-json",
        dest="candidates_json",
        metavar="PATH",
        help="Optional candidates JSON; omit to build from bundled data/natural25/scene_events_25x4.csv.",
    )
    p_task.add_argument(
        "--families-jsonl",
        dest="families_jsonl",
        metavar="PATH",
        help="Optional families JSONL; omit to use bundled data/natural25/families.jsonl.",
    )
    p_task.add_argument("--output", metavar="PATH")

    p_ff = sub.add_parser("firstframe", help="Generate first-frame images.")
    p_ff.add_argument("--out", required=True, metavar="DIR", help="Output directory for PNGs.")
    p_ff.add_argument("--families-jsonl", dest="families_jsonl", metavar="PATH")
    p_ff.add_argument("--family-id", dest="family_id", metavar="ID")
    p_ff.add_argument("--prompt", metavar="TEXT", help="T2I prompt (with --family-id).")
    p_ff.add_argument("--provider", required=True, help="T2I provider: dashscope, mock.")
    p_ff.add_argument("--model", required=True, help="T2I model name.")
    p_ff.add_argument("--api-key", dest="api_key", help="T2I API key.")
    p_ff.add_argument("--endpoint", required=True, help="T2I API endpoint.")
    p_ff.add_argument("--size", required=True, help="T2I output size, provider-specific.")
    p_ff.add_argument("--n", required=True, help="T2I output count, provider-specific.")
    ff_existing = p_ff.add_mutually_exclusive_group(required=True)
    ff_existing.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing PNGs and record them in the manifest.",
    )
    ff_existing.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Regenerate existing PNGs.",
    )
