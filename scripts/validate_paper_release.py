#!/usr/bin/env python
"""Validate a bundled or staged WRBench paper-release directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_src() -> Path:
    return Path(__file__).resolve().parents[1] / "src"


if str(_project_src()) not in sys.path:
    sys.path.insert(0, str(_project_src()))

from wrbench.datasets import NATURAL25_PAPER_RELEASE_ID  # noqa: E402
from wrbench.release_validation import validate_natural25_release  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", default=NATURAL25_PAPER_RELEASE_ID)
    parser.add_argument(
        "--release-dir",
        type=Path,
        help="Validate an explicit staged directory instead of bundled package data.",
    )
    args = parser.parse_args(argv)
    summary = validate_natural25_release(args.release_id, release_dir=args.release_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
