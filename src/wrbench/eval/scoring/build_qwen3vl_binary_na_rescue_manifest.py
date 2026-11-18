#!/usr/bin/env python3
"""Build a guarded-rescue manifest from Qwen3-VL binary gate N/A evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from wrbench.eval._io import read_json, read_records, write_json
from wrbench.eval.scoring.runtime_common import index_by_video_id

SCHEMA_VERSION = "qwen3vl_binary_na_rescue_manifest_v1"
load_json = read_json
load_jsonl = read_records


def _manifest_ids(manifest: list[dict[str, Any]]) -> list[str]:
    indexed = index_by_video_id(manifest, label="manifest", require_unique=True)
    return list(indexed)


def _gate_bool(row: dict[str, Any], video_id: str, key: str) -> bool:
    value = row.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"non-boolean {key} for video_id={video_id}: {value!r}")
    return value


def build_rescue_manifest(
    *,
    manifest: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = _manifest_ids(manifest)
    manifest_by_id = {str(row["video_id"]): row for row in manifest}
    evidence_by_id = index_by_video_id(evidence_rows, label="evidence", require_unique=True)
    missing = [video_id for video_id in manifest_ids if video_id not in evidence_by_id]
    if missing:
        raise ValueError(f"missing evidence for {len(missing)} manifest videos: {missing[:10]}")

    extra = sorted(set(evidence_by_id) - set(manifest_ids))
    selected_ids: list[str] = []
    selected_reasons: dict[str, list[str]] = {}
    request_mapping: list[dict[str, Any]] = []
    dim_counter: Counter[str] = Counter()
    for video_id in manifest_ids:
        manifest_row = manifest_by_id[video_id]
        evidence = evidence_by_id[video_id]
        reasons: list[str] = []
        if not _gate_bool(evidence, video_id, "evidence_d5_applicable"):
            reasons.append("D5")
            dim_counter["D5"] += 1
        if not _gate_bool(evidence, video_id, "evidence_d6_applicable"):
            reasons.append("D6")
            dim_counter["D6"] += 1
        if reasons:
            selected_ids.append(video_id)
            selected_reasons[video_id] = reasons
            request_mapping.append(
                {
                    "video_id": video_id,
                    "request_id": manifest_row.get("request_id"),
                    "row_index": manifest_row.get("row_index"),
                    "evidence_request_id": evidence.get("request_id"),
                    "evidence_row_index": evidence.get("row_index"),
                    "reasons": reasons,
                }
            )

    rescue_manifest = [dict(manifest_by_id[video_id]) for video_id in selected_ids]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "manifest_records": len(manifest_ids),
        "evidence_records": len(evidence_rows),
        "selected_records": len(rescue_manifest),
        "false_dimension_counts": {dim: int(dim_counter.get(dim, 0)) for dim in ("D5", "D6")},
        "selected_video_ids": selected_ids,
        "selected_request_ids": [row.get("request_id") for row in request_mapping],
        "selected_row_indices": [row.get("row_index") for row in request_mapping],
        "selected_reasons_by_video_id": selected_reasons,
        "request_mapping": request_mapping,
        "extra_evidence_video_ids": extra,
    }
    return rescue_manifest, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--binary-evidence-jsonl", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise TypeError("--manifest-path must contain a JSON list")
    rescue_manifest, summary = build_rescue_manifest(
        manifest=manifest,
        evidence_rows=read_records(args.binary_evidence_jsonl),
    )
    summary.update(
        {
            "manifest_path": str(args.manifest_path),
            "binary_evidence_jsonl": str(args.binary_evidence_jsonl),
            "out_manifest": str(args.out_manifest),
            "out_summary": str(args.out_summary),
        }
    )
    write_json(args.out_manifest, rescue_manifest, sort_keys=True)
    write_json(args.out_summary, summary, sort_keys=True)
    print(f"[wrote] {args.out_manifest}")
    print(f"[wrote] {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
