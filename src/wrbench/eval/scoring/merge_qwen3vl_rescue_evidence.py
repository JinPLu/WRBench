#!/usr/bin/env python3
"""Overlay Qwen3-VL rescue evidence onto complete baseline evidence."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from wrbench.eval._io import read_json, read_records, write_json, write_jsonl
from wrbench.eval.scoring.runtime_common import index_by_video_id

SCHEMA_VERSION = "qwen3vl_rescue_evidence_overlay_v1"
load_json = read_json
load_jsonl = read_records


def overlay_evidence(
    *,
    manifest: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    rescue_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = index_by_video_id(baseline_rows, label="baseline", require_unique=True)
    rescue_by_id = index_by_video_id(rescue_rows, label="rescue", require_unique=True)
    manifest_ids = list(index_by_video_id(manifest, label="manifest", require_unique=True))
    missing_baseline = [video_id for video_id in manifest_ids if video_id not in baseline_by_id]
    output_rows: list[dict[str, Any]] = []
    replaced_ids: list[str] = []
    request_mapping: list[dict[str, Any]] = []
    for video_id in manifest_ids:
        row = rescue_by_id.get(video_id) or baseline_by_id.get(video_id)
        if row is None:
            request_mapping.append(
                {
                    "video_id": video_id,
                    "request_id": None,
                    "row_index": None,
                    "baseline_present": False,
                    "rescue_present": False,
                    "overlay_source": None,
                }
            )
            continue
        out = dict(row)
        if video_id in rescue_by_id:
            out["evidence_overlay_source"] = "guarded_rescue"
            replaced_ids.append(video_id)
        else:
            out["evidence_overlay_source"] = "binary_baseline"
        output_rows.append(out)
        request_mapping.append(
            {
                "video_id": video_id,
                "request_id": out.get("request_id"),
                "row_index": out.get("row_index"),
                "baseline_present": video_id in baseline_by_id,
                "rescue_present": video_id in rescue_by_id,
                "overlay_source": out["evidence_overlay_source"],
            }
        )
    unused_rescue_ids = sorted(set(rescue_by_id) - set(manifest_ids))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_records": len(manifest_ids),
        "baseline_records": len(baseline_by_id),
        "rescue_records": len(rescue_by_id),
        "output_records": len(output_rows),
        "replaced_records": len(replaced_ids),
        "replaced_video_ids": replaced_ids,
        "missing_baseline_video_ids": missing_baseline,
        "unused_rescue_video_ids": unused_rescue_ids,
        "request_mapping": request_mapping,
    }
    return output_rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--baseline-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--rescue-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--out-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise TypeError("--manifest-path must contain a JSON list")
    output_rows, summary = overlay_evidence(
        manifest=manifest,
        baseline_rows=read_records(args.baseline_evidence_jsonl),
        rescue_rows=read_records(args.rescue_evidence_jsonl),
    )
    write_jsonl(output_rows, args.out_evidence_jsonl, sort_keys=True)
    summary.update(
        {
            "manifest_path": str(args.manifest_path),
            "baseline_evidence_jsonl": str(args.baseline_evidence_jsonl),
            "rescue_evidence_jsonl": str(args.rescue_evidence_jsonl),
            "out_evidence_jsonl": str(args.out_evidence_jsonl),
        }
    )
    write_json(args.out_summary_json, summary, sort_keys=True)
    print(f"[wrote] {args.out_evidence_jsonl}")
    print(f"[wrote] {args.out_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
