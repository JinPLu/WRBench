#!/usr/bin/env python
"""Prepare the pinned 100 Wan2.7 static source videos for the paper release."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _project_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


if str(_project_src()) not in sys.path:
    sys.path.insert(0, str(_project_src()))

from wrbench.datasets import (  # noqa: E402
    NATURAL25_PAPER_RELEASE_ID,
    natural25_release_tv2v_sources_path,
)
from wrbench.paper_sources import prepare_paper_tv2v_sources  # noqa: E402
from wrbench.release_validation import (  # noqa: E402
    load_tv2v_source_rows,
    validate_natural25_release,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=NATURAL25_PAPER_RELEASE_ID)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Override the bundled pinned source manifest (primarily for audited staging).",
    )
    parser.add_argument(
        "--source-video-root",
        required=True,
        type=Path,
        help="Local root under which immutable source assets and the task map are materialized.",
    )
    parser.add_argument(
        "--task-map",
        type=Path,
        help="Output task map path; defaults to SOURCE_ROOT/source_video_task_map.json.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", "--dry-run", dest="plan", action="store_true", help="Print the plan without writes.")
    mode.add_argument("--download", action="store_true", help="Download, checksum, and write the task map.")
    mode.add_argument("--verify-only", action="store_true", help="Verify local files without writes.")
    parser.add_argument("--resume", action="store_true", help="Resume .part downloads and reuse verified files.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Replace corrupt/conflicting local files explicitly; otherwise overwrite is refused.",
    )
    args = parser.parse_args(argv)

    if args.manifest is None:
        validate_natural25_release(args.release_id)
        manifest_path = natural25_release_tv2v_sources_path(args.release_id)
    else:
        manifest_path = args.manifest.expanduser().resolve()
    rows = load_tv2v_source_rows(manifest_path)
    selected_mode = "plan" if args.plan else "download" if args.download else "verify-only"
    summary = prepare_paper_tv2v_sources(
        rows,
        release_id=args.release_id,
        source_root=args.source_video_root,
        task_map_path=args.task_map,
        mode=selected_mode,
        resume=args.resume,
        overwrite_existing=args.overwrite_existing,
    )
    statuses = Counter(item["status"] for item in summary["results"])
    print(
        json.dumps(
            {
                "mode": summary["mode"],
                "release_id": summary["release_id"],
                "rows": summary["rows"],
                "source_root": summary["source_root"],
                "status_counts": dict(sorted(statuses.items())),
                "task_map": summary["task_map"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
