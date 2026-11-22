"""CLI subcommand: generate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from wrbench._cli.common import add_model_spec_argument


def _resolve_preset_camera(camera_str: str, args: argparse.Namespace):
    """Return a CameraScript or camera string for ``--camera preset:<name>``."""
    import wrbench

    if not camera_str.startswith("preset:"):
        return camera_str  # raw script string

    preset_name = camera_str[len("preset:") :]
    kwargs: dict[str, Any] = {}
    if args.peak_deg is not None:
        kwargs["peak_deg"] = args.peak_deg
    if args.amount is not None:
        kwargs["amount"] = args.amount
    if args.frames is not None:
        kwargs["frames"] = args.frames

    try:
        script = wrbench.presets.build_preset(preset_name, **kwargs)
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        names = wrbench.presets.preset_names()
        print(f"  valid presets: {', '.join(names)}", file=sys.stderr)
        raise SystemExit(1)
    return script


def _cmd_generate(args: argparse.Namespace) -> int:
    import wrbench

    model_name: str = args.model
    camera_arg: str = args.camera
    out_path: str = args.out
    dry_run: bool = not args.no_dry_run

    # Load runtime config early so top-level model_spec_files are registered before model lookup
    runtime: Any = None
    if args.runtime_config or args.no_dry_run:
        from wrbench.runtime import RuntimeConfigError, load_runtime_config

        try:
            runtime = (
                load_runtime_config(Path(args.runtime_config))
                if args.runtime_config
                else load_runtime_config()
            )
        except (RuntimeConfigError, OSError, ValueError) as exc:
            print(f"error: failed to load runtime config: {exc}", file=sys.stderr)
            return 1

    # Resolve model key early so errors are clear
    try:
        key = wrbench.canonical_model_key(model_name)
    except KeyError:
        known = wrbench.list_models(include_deferred=True)
        print(f"error: unknown model {model_name!r}", file=sys.stderr)
        print(f"  known models: {', '.join(known)}", file=sys.stderr)
        return 1

    # Check deferred status
    record = wrbench.model_record(key)
    if record.is_deferred:
        print(
            f"error: model {key!r} is deferred (not yet dispatchable). "
            "Generation is not supported for deferred models.",
            file=sys.stderr,
        )
        return 1

    # Resolve camera
    camera = _resolve_preset_camera(camera_arg, args)

    # Validate inputs based on input_kind
    input_kind = record.input_kind
    image_path: str | None = args.image
    source_video_path: str | None = args.source_video

    if input_kind == "image":
        if source_video_path:
            print(
                f"error: model {key!r} is an image/TI2V model; use --image, not --source-video.",
                file=sys.stderr,
            )
            return 1
        if not image_path:
            print(
                f"error: model {key!r} is an image/TI2V model and requires --image <path>.",
                file=sys.stderr,
            )
            return 1
    elif input_kind == "source_video":
        if image_path:
            print(
                f"error: model {key!r} uses a source-video runtime wrapper; "
                "use --source-video, not --image.",
                file=sys.stderr,
            )
            return 1
        if not source_video_path:
            print(
                f"error: model {key!r} uses a source-video runtime wrapper and requires "
                f"--source-video <path> (model_input={record.model_input}, "
                f"source_video_usage={record.source_video_usage}).",
                file=sys.stderr,
            )
            return 1
    elif input_kind == "none":
        if image_path or source_video_path:
            print(
                f"error: model {key!r} does not use --image or --source-video; use --prompt for text conditioning.",
                file=sys.stderr,
            )
            return 1
        if args.no_dry_run and not str(args.prompt).strip():
            print(
                f"error: model {key!r} is prompt-only and requires --prompt for --no-dry-run.",
                file=sys.stderr,
            )
            return 1

    # Pre-check backend availability for ALL input kinds when --no-dry-run is requested
    if args.no_dry_run:
        from wrbench.backends.registry import resolve_backend
        from wrbench.runtime import RuntimeConfigError

        try:
            backend = resolve_backend(key, runtime=runtime)
        except (RuntimeConfigError, ValueError, KeyError, ImportError, RuntimeError) as exc:
            print(f"error: failed to resolve generation backend for {key!r}: {exc}", file=sys.stderr)
            return 1

        if getattr(backend, "name", "") == "dry_run":
            print(
                f"error: --no-dry-run requested but only the dry-run backend is available for {key!r}.",
                file=sys.stderr,
            )
            print(
                "  Configure wrbench.runtime.json (see wrbench.runtime.example.json) "
                "or omit --no-dry-run to compile sidecars only.",
                file=sys.stderr,
            )
            return 1

        if hasattr(backend, "available_for"):
            ok, reason = backend.available_for(key)  # type: ignore[attr-defined]
        else:
            ok, reason = backend.available()
        if not ok:
            print(
                f"error: --no-dry-run requested but no real backend is available for {key!r}: {reason}",
                file=sys.stderr,
            )
            print(
                "  Configure wrbench.runtime.json (see wrbench.runtime.example.json) "
                "or omit --no-dry-run to compile sidecars only.",
                file=sys.stderr,
            )
            return 1
        dry_run = False

    try:
        result = wrbench.compile_camera(
            model=key,
            camera=camera,
            out=out_path,
            image=image_path,
            source_video=source_video_path,
            prompt=args.prompt,
            width=args.width,
            height=args.height,
            fps=args.fps,
            num_frames=args.num_frames,
            work_dir=args.work_dir,
            runtime_config=args.runtime_config,
            dry_run=dry_run,
        )
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    artifacts: dict[str, str] = result.get("artifacts", {})
    print(f"model    : {result['model']}")
    print(f"dry_run  : {result['dry_run']}")
    print("artifacts:")
    for name, path in artifacts.items():
        exists_mark = " (written)" if Path(path).exists() else " (missing)"
        print(f"  {name}: {path}{exists_mark}")
    generation = result.get("generation")
    if generation:
        print(f"generation: success={generation.get('success')} message={generation.get('message')}")
        if generation.get("output_path"):
            print(f"  output: {generation['output_path']}")
    return 0


def register_generate_parser(sub: argparse._SubParsersAction[Any]) -> None:
    """Register the generate subparser."""
    p_gen = sub.add_parser(
        "generate", help="Compile a camera script and write model-native payload sidecars."
    )
    p_gen.add_argument("--model", required=True, metavar="MODEL", help="Target model key.")
    p_gen.add_argument(
        "--camera",
        required=True,
        metavar="SCRIPT",
        help=(
            'Camera script or "preset:<name>", e.g. "yaw:left:60@40" or "preset:yaw_LR". '
            "Combine with --peak-deg, --amount, --frames to parametrize presets."
        ),
    )
    p_gen.add_argument("--out", required=True, metavar="PATH", help="Output file path (e.g. out.mp4).")
    p_gen.add_argument("--image", metavar="PATH", help="Input image (required for TI2V models).")
    p_gen.add_argument(
        "--source-video",
        metavar="PATH",
        dest="source_video",
        help="Input source video required by source-video runtime wrappers.",
    )
    p_gen.add_argument("--prompt", required=True, metavar="TEXT", help="Text prompt.")
    p_gen.add_argument("--width", type=int, metavar="N", help="Override model registry frame width.")
    p_gen.add_argument("--height", type=int, metavar="N", help="Override model registry frame height.")
    p_gen.add_argument("--fps", type=int, metavar="N", help="Override model registry frames per second.")
    p_gen.add_argument(
        "--num-frames",
        type=int,
        dest="num_frames",
        metavar="N",
        help="Override total frame count.",
    )
    p_gen.add_argument("--work-dir", metavar="DIR", dest="work_dir", help="Working directory for scratch files.")
    p_gen.add_argument(
        "--runtime-config",
        metavar="PATH",
        dest="runtime_config",
        help="Explicit wrbench.runtime.json path for real generation.",
    )
    p_gen.add_argument(
        "--no-dry-run",
        action="store_true",
        dest="no_dry_run",
        help=(
            "Disable dry-run mode and invoke the configured real-generation backend. "
            "Requires wrbench.runtime.json."
        ),
    )
    p_gen.add_argument(
        "--peak-deg",
        type=float,
        dest="peak_deg",
        metavar="DEG",
        help="Override preset peak_deg (yaw/pitch presets).",
    )
    p_gen.add_argument(
        "--amount",
        type=float,
        metavar="A",
        help="Override preset amount (translation presets).",
    )
    p_gen.add_argument(
        "--frames",
        type=int,
        metavar="N",
        help="Override preset total frame count.",
    )
    add_model_spec_argument(p_gen)
