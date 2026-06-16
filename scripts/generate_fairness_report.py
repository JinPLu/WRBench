#!/usr/bin/env python3
"""Build fairness + capability verification report from acceptance artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
import sys

sys.path.insert(0, str(SRC))

from wrcam.registry import active_model_keys, model_record  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_acceptance_csv(path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            model = str(row.get("model") or "").strip()
            if model:
                rows[model] = row
    return rows


def fairness_gate(row: dict[str, str]) -> str:
    decision = row.get("final_decision") or row.get("decision") or ""
    direction = row.get("direction_ok") or row.get("direction_ok_rate") or ""
    frame_clean = row.get("frame_clean") or row.get("frame_clean_rate") or ""
    if decision in {"BLOCKED", "NO_VIDEOS"}:
        return decision
    try:
        direction_f = float(direction) if direction not in ("", "None", None) else None
    except ValueError:
        direction_f = None
    try:
        frame_f = float(frame_clean) if frame_clean not in ("", "None", None) else None
    except ValueError:
        frame_f = None
    if frame_f is not None and frame_f < 1.0:
        return "frame_qc_fail"
    if direction_f is not None and direction_f >= 0.8:
        return "fair_direction_pass"
    if direction_f is not None and direction_f < 0.5:
        return "direction_weak"
    if decision == "ACCEPT":
        return "fair_direction_pass"
    return "advisory_amplitude_only"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--acceptance-csv",
        type=Path,
        default=REPO.parent / "WRBench/WRBenchLib/.artifacts/camera_calibration_acceptance/ACCEPTANCE_SUMMARY.csv",
    )
    parser.add_argument("--out", type=Path, default=REPO / "docs/data/fairness_verification_report.md")
    args = parser.parse_args(argv)

    if not args.acceptance_csv.is_file():
        fallback = REPO / "docs/data/acceptance_summary.csv"
        if fallback.is_file():
            args.acceptance_csv = fallback
        else:
            raise FileNotFoundError(args.acceptance_csv)

    acceptance = load_acceptance_csv(args.acceptance_csv)
    lines = [
        "# Fairness and capability verification report\n",
        f"Generated: {utc_now()}\n\n",
        "Policy: **fair comparison + full capability utilization**, not perfect metrics.\n\n",
        "- **Fairness gate**: VGGT-Omega/D1 direction-correct + frame decode clean (contact-sheet auto QC).\n",
        "- **Advisory only**: D1 amplitude / mean accuracy (`NEEDS_CALIBRATION` ≠ automatic fail).\n",
        "- **Capability**: native payload faithfulness via WRCam sidecar taxonomy (`certification_kind`, `target_c2w_is_model_effective`).\n\n",
        "| model | fairness_gate | final_decision | direction_ok | frame_clean | d1_mean (advisory) | certification_kind | target_c2w_is_model_effective |\n",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |\n",
    ]

    for key in sorted(set(active_model_keys()) | set(acceptance.keys())):
        row = acceptance.get(key, {})
        try:
            record = model_record(key)
            meta = record.amplitude.metadata or {}
            cert = "compile-time in adapter"
            effective = meta.get("target_c2w_is_model_effective", "n/a")
        except KeyError:
            cert = "n/a"
            effective = "n/a"
        gate = fairness_gate(row) if row else "no_acceptance_data"
        lines.append(
            f"| {key} | {gate} | {row.get('final_decision','-')} | "
            f"{row.get('direction_ok', row.get('direction_ok_rate','-'))} | "
            f"{row.get('frame_clean', row.get('frame_clean_rate','-'))} | "
            f"{row.get('d1_mean', row.get('mean_d1_camera_accuracy','-'))} | {cert} | {effective} |\n"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(lines), encoding="utf-8")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
