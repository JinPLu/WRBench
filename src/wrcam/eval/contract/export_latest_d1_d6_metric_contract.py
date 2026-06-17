#!/usr/bin/env python3
"""Export the public WRBench D1-D6 metric contract into this directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wrcam.eval.aggregate.latest_d1_d6_metrics import write_metric_contract


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    out_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=out_dir / "latest_d1_d6_metric_contract.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=out_dir / "latest_d1_d6_metric_contract.md",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    write_metric_contract(out_json=args.out_json, out_md=args.out_md)
    print(f"[wrote] {args.out_json}")
    print(f"[wrote] {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
