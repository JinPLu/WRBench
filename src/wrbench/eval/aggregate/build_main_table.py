#!/usr/bin/env python3
"""Build the WRBench D1-D6 main table from the canonical metric contract."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from wrbench.eval._io import read_records, write_json
from wrbench.eval.aggregate import latest_d1_d6_metrics as metric_contract
from wrbench.eval.d1.nominal_target import camera_type, is_prompt_only_row
from wrbench.registry import RegistryError, model_input as registry_model_input, viewpoint_condition_type

EXCLUDED_MAIN_TABLE_MODELS = set(
    metric_contract.MAIN_TABLE_HYPERPARAMETERS["excluded_main_table_models"]
)
DIMENSION_SPECS = metric_contract.latest_metric_specs()
D1_SPEC = metric_contract.metric_by_dimension("D1")
D2_SPEC = metric_contract.metric_by_dimension("D2")
GATE_FIELD = metric_contract.WORLDSTATE_GATE_FIELD
NOMINAL_D1_COLUMN = "D1_camera_pose_nominal"
STATIC_HOLD_COLUMN = "D1_camalign_static_hold"
CAMALIGN_SPEC = metric_contract.metric_by_dimension("D1-CamAlign")


def _viewpoint_condition(
    model: str,
    rows: Mapping[str, Any] | list[dict[str, Any]] | None = None,
) -> str:
    try:
        value = viewpoint_condition_type(model)
        if value:
            return value
    except (KeyError, RegistryError):
        pass
    candidates: list[Mapping[str, Any]] = []
    if isinstance(rows, Mapping):
        candidates = [rows]
    elif rows:
        candidates = list(rows)
    for row in candidates:
        for field in ("viewpoint_condition_type", "viewpoint_condition"):
            raw = row.get(field)
            if raw not in (None, ""):
                return str(raw)
        if is_prompt_only_row(row):
            return "prompt-only"
    return "custom"


def _model_input(
    model: str,
    rows: Mapping[str, Any] | list[dict[str, Any]] | None = None,
) -> str:
    try:
        value = registry_model_input(model)
        if value:
            return value
    except (KeyError, RegistryError):
        pass
    candidates: list[Mapping[str, Any]] = []
    if isinstance(rows, Mapping):
        candidates = [rows]
    elif rows:
        candidates = list(rows)
    for row in candidates:
        raw = row.get("model_input")
        if raw not in (None, ""):
            return str(raw)
    return "unknown"


def _gate_applicable(row: Mapping[str, Any]) -> bool | None:
    for field in (GATE_FIELD, "runtime_v2_shared_oov_gate_applicable"):
        if field in row:
            value = row.get(field)
            if value in (None, ""):
                return None
            if isinstance(value, bool):
                return value
            if str(value).strip().lower() in {"true", "1", "false", "0"}:
                return str(value).strip().lower() in {"true", "1"}
            raise ValueError(f"{field} must be boolean for video_id={row.get('video_id')}")
    return None


def _load_records(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    return read_records(path)


def _index(records: list[dict[str, Any]], *, artifact_name: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for offset, row in enumerate(records, start=1):
        video_id = row.get("video_id")
        if not video_id:
            raise ValueError(f"{artifact_name} row {offset}: missing video_id")
        key = str(video_id)
        if key in indexed:
            raise ValueError(f"{artifact_name}: duplicate video_id {key}")
        indexed[key] = row
    return indexed


def _index_by_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        model = row.get("model")
        if model and model not in EXCLUDED_MAIN_TABLE_MODELS:
            buckets[str(model)].append(row)
    return buckets


def build_table(
    *,
    runtime_records: list[dict[str, Any]],
    d1_records: list[dict[str, Any]],
    d2_records: list[dict[str, Any]],
    d1_camalign_records: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _index(runtime_records, artifact_name="D3–D6 score artifact")
    d1_by_id = _index(d1_records, artifact_name="D1 score artifact")
    d2_by_id = _index(d2_records, artifact_name="D2 score artifact")
    camalign_by_id = (
        _index(d1_camalign_records, artifact_name="D1 CamAlign artifact")
        if d1_camalign_records is not None
        else {}
    )
    d1_by_model = _index_by_model(d1_records)
    d2_by_model = _index_by_model(d2_records)
    camalign_by_model = _index_by_model(d1_camalign_records) if d1_camalign_records is not None else {}
    buckets = _index_by_model(runtime_records)

    table: list[dict[str, Any]] = []
    runtime_models = set(buckets)
    d1_models = set(d1_by_model)
    d2_models = set(d2_by_model)
    camalign_models = set(camalign_by_model)
    models_without_d2_d3_d6_artifacts: set[str] = set()
    required_specs = (D1_SPEC, D2_SPEC)
    if d1_camalign_records is not None:
        required_specs += (CAMALIGN_SPEC,)
    for model in sorted(runtime_models | d1_models | camalign_models | d2_models):
        rows = (
            buckets.get(model)
            or d1_by_model.get(model)
            or camalign_by_model.get(model)
            or d2_by_model.get(model, [])
        )
        model_vct = _viewpoint_condition(model, rows)
        model_inp = _model_input(model, rows)
        out: dict[str, Any] = {
            "model": model,
            "n_records": len(rows),
            "viewpoint_condition_type": model_vct,
            "model_input": model_inp,
        }
        ready = True
        gates = {str(row["video_id"]): _gate_applicable(row) for row in buckets.get(model, ())}
        gate_values = [gate for gate in gates.values() if gate is not None]
        nominal_values: list[float] = []
        static_hold_values: list[float] = []
        static_hold_eligible = 0
        for spec in DIMENSION_SPECS:
            column = spec.output_column
            values: list[float] = []
            eligible = 0
            for row in rows:
                video_id = str(row.get("video_id") or "")
                if spec.dimension_id == "D1":
                    camera = camera_type(row)
                    if camera == "uncontrolled" or (model == "hunyuan_game_craft" and camera == "static"):
                        continue
                    d1_row = d1_by_id.get(video_id)
                    if (
                        model_vct == "prompt-only"
                        or is_prompt_only_row(row)
                        or str(row.get("D1_CamPrec_applicable")).strip().lower() == "false"
                        or (
                            d1_row is not None
                            and str(d1_row.get("d1_status") or "")
                            in {"excluded_uncontrolled", "excluded_prompt_only"}
                        )
                    ):
                        continue
                    eligible += 1
                    if (
                        d1_row
                        and d1_row.get("d1_status") == "ok"
                        and d1_row.get("d1_target_source") == "nominal_profile_fallback"
                        and d1_row.get("d1_nominal_camera_accuracy") is not None
                    ):
                        nominal_values.append(float(d1_row["d1_nominal_camera_accuracy"]))
                    value = (
                        metric_contract.require_metric_value(d1_row, spec, context=f"D1 score row video_id={video_id}")
                        if d1_row
                        else None
                    )
                elif spec.dimension_id == "D1-CamAlign":
                    camalign_row = camalign_by_id.get(video_id)
                    camera = camera_type(row) or camera_type(camalign_row or {})
                    is_static_hold = camera == "static" and (
                        model_vct == "prompt-only" or is_prompt_only_row(camalign_row or row)
                    )
                    if camera not in {"yaw_LR", "yaw_RL"} and not is_static_hold:
                        continue
                    value = (
                        metric_contract.require_metric_value(
                            camalign_row,
                            spec,
                            context=f"D1 CamAlign row video_id={video_id}",
                        )
                        if camalign_row
                        else None
                    )
                    if is_static_hold:
                        static_hold_eligible += 1
                        if value is not None:
                            static_hold_values.append(value)
                        continue
                    eligible += 1
                elif spec.dimension_id == "D2":
                    if video_id not in d2_by_id and model not in runtime_models:
                        continue
                    eligible += 1
                    d2_row = d2_by_id.get(video_id)
                    if d2_row is None or ("d2_status" in d2_row and d2_row.get("d2_status") != "ok"):
                        value = None
                    else:
                        value = metric_contract.require_metric_value(
                            d2_row,
                            spec,
                            context=f"D2 score row video_id={video_id}",
                        )
                else:
                    if model not in runtime_models:
                        continue
                    gate = gates.get(video_id)
                    is_returned = spec.dimension_id in {"D5", "D6"}
                    if is_returned:
                        if gate is None:
                            raise ValueError(f"D3–D6 score row video_id={video_id}: missing shared re-observation gate")
                        if not gate:
                            continue
                    eligible += 1
                    value = metric_contract.require_metric_value(
                        row,
                        spec,
                        context=f"{spec.dimension_id} runtime row video_id={video_id}",
                    )
                    if is_returned and value is None:
                        raise ValueError(f"{spec.dimension_id} runtime row video_id={video_id}: applicable score is null")
                if value is not None:
                    values.append(value)
            out[column] = mean(values) if values else None
            out[f"{column}_n"] = len(values)
            out[f"{column}_eligible"] = eligible
            if spec in required_specs and len(values) < eligible:
                ready = False
            if spec.dimension_id == "D6" and model not in runtime_models:
                models_without_d2_d3_d6_artifacts.add(model)
        out[NOMINAL_D1_COLUMN] = mean(nominal_values) if nominal_values else None
        out[f"{NOMINAL_D1_COLUMN}_n"] = len(nominal_values)
        out[STATIC_HOLD_COLUMN] = mean(static_hold_values) if static_hold_values else None
        out[f"{STATIC_HOLD_COLUMN}_n"] = len(static_hold_values)
        out[f"{STATIC_HOLD_COLUMN}_eligible"] = static_hold_eligible
        if gate_values:
            out["reobservation_support"] = sum(1 for v in gate_values if v) / len(gate_values)
            out["reobservation_support_n"] = len(gate_values)
        else:
            out["reobservation_support"] = None
            out["reobservation_support_n"] = 0
        out["main_table_ready"] = ready
        table.append(out)

    main_table_ready = bool(table) and all(row["main_table_ready"] for row in table)
    pending_evidence_dimensions = list(metric_contract.pending_evidence_dimensions())
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric_contract_schema_version": metric_contract.SCHEMA_VERSION,
        "metric_source_fields": {spec.dimension_id: spec.score_field for spec in DIMENSION_SPECS},
        "forbidden_fallback_fields": {
            spec.dimension_id: list(spec.forbidden_fallback_fields) for spec in DIMENSION_SPECS
        },
        "metric_hyperparameters": {
            spec.dimension_id: dict(spec.hyperparameters or {}) for spec in DIMENSION_SPECS
        },
        "metric_evidence_status": metric_contract.metric_evidence_status(),
        "pending_evidence_dimensions": pending_evidence_dimensions,
        "table_hyperparameters": dict(metric_contract.MAIN_TABLE_HYPERPARAMETERS),
        "runtime_records": len(runtime_records),
        "d1_records": len(d1_records),
        "d2_records": len(d2_records),
        "usable_models": len(table),
        "main_table_ready": main_table_ready,
        "reviewed_current_truth_ready": main_table_ready and not pending_evidence_dimensions,
        "runtime_models": sorted(runtime_models),
        "d1_models_without_runtime_records": sorted(d1_models - runtime_models),
        "d2_models_without_runtime_records": sorted(d2_models - runtime_models),
        "models_without_d2_d3_d6_artifacts": sorted(models_without_d2_d3_d6_artifacts),
        "qa_notes": [
            "D1-D6 output columns and source fields come from latest_d1_d6_metrics.py.",
            "Nominal D1 target comparisons do not contribute to strict CamPrec.",
            "CamAlign common-yaw includes local and API yaw. API static-hold is separate.",
            "D2 accepts only d2_selected_visual_integrity_score; d2_dinov2_temporal_consistency is forbidden as a fallback.",
            "D2 accepts rows without d2_status; when d2_status is present it must be ok.",
        ],
        "blocked_dimensions": [
            spec.output_column
            for spec in required_specs
            if sum(row[f"{spec.output_column}_n"] for row in table)
            < sum(row[f"{spec.output_column}_eligible"] for row in table)
        ],
    }
    return table, summary


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "viewpoint_condition_type",
        "model_input",
        "n_records",
        "reobservation_support",
        "reobservation_support_n",
    ]
    for spec in DIMENSION_SPECS:
        fields.extend([spec.output_column, f"{spec.output_column}_n"])
        if spec in (D1_SPEC, D2_SPEC, CAMALIGN_SPEC):
            fields.append(f"{spec.output_column}_eligible")
        if spec == D1_SPEC:
            fields.extend([NOMINAL_D1_COLUMN, f"{NOMINAL_D1_COLUMN}_n"])
        elif spec == CAMALIGN_SPEC:
            fields.extend([STATIC_HOLD_COLUMN, f"{STATIC_HOLD_COLUMN}_n", f"{STATIC_HOLD_COLUMN}_eligible"])
    fields.append("main_table_ready")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_md(path: Path, rows: list[dict[str, Any]], summary: Mapping[str, Any]) -> None:
    lines = [
        "# WRBench Main Table",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Metric contract: `{summary['metric_contract_schema_version']}`",
        f"- Main table ready: `{summary['main_table_ready']}`",
        f"- Reviewed current truth ready: `{summary['reviewed_current_truth_ready']}`",
        f"- Blocked dimensions: `{', '.join(summary['blocked_dimensions']) or 'none'}`",
        f"- Pending evidence dimensions: `{', '.join(summary['pending_evidence_dimensions']) or 'none'}`",
        f"- Metric source fields: `{json.dumps(summary['metric_source_fields'], sort_keys=True)}`",
        f"- Forbidden fallback fields: `{json.dumps(summary['forbidden_fallback_fields'], sort_keys=True)}`",
        f"- D1-only / no runtime models: `{', '.join(summary['d1_models_without_runtime_records']) or 'none'}`",
        f"- Models without D2/D3-D6 artifacts: `{', '.join(summary['models_without_d2_d3_d6_artifacts']) or 'none'}`",
        "",
        "## QA Notes",
        "",
        *[f"- {note}" for note in summary["qa_notes"]],
        "",
        "## Table",
        "",
        "| model | n | D1 pose | D1 nominal | CamAlign yaw | API static | D2 visual | D3 spatial-in | D4 state-in | D5 spatial-OoV | D6 state-OoV | ready |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["model"]),
                    str(row["n_records"]),
                    f"{_fmt(row['D1_camera_pose'])} ({row['D1_camera_pose_n']}/{row['D1_camera_pose_eligible']})",
                    f"{_fmt(row[NOMINAL_D1_COLUMN])} ({row[f'{NOMINAL_D1_COLUMN}_n']})",
                    f"{_fmt(row['D1_camalign'])} ({row['D1_camalign_n']}/{row['D1_camalign_eligible']})",
                    f"{_fmt(row[STATIC_HOLD_COLUMN])} ({row[f'{STATIC_HOLD_COLUMN}_n']}/{row[f'{STATIC_HOLD_COLUMN}_eligible']})",
                    f"{_fmt(row['D2_visual_integrity'])} ({row['D2_visual_integrity_n']}/{row['D2_visual_integrity_eligible']})",
                    f"{_fmt(row['D3_spatial_in'])} ({row['D3_spatial_in_n']})",
                    f"{_fmt(row['D4_state_in'])} ({row['D4_state_in_n']})",
                    f"{_fmt(row['D5_spatial_oov'])} ({row['D5_spatial_oov_n']})",
                    f"{_fmt(row['D6_state_oov'])} ({row['D6_state_oov_n']})",
                    str(row["main_table_ready"]),
                ]
            )
            + " |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--d3d6-scores",
        "--vlm-scores",
        "--runtime-scores",
        dest="runtime_scores",
        type=Path,
        required=True,
    )
    parser.add_argument("--d1-scores", type=Path)
    parser.add_argument("--d1-camalign-scores", type=Path)
    parser.add_argument("--d2-scores", type=Path)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    table, summary = build_table(
        runtime_records=_load_records(args.runtime_scores),
        d1_records=_load_records(args.d1_scores),
        d2_records=_load_records(args.d2_scores),
        d1_camalign_records=_load_records(args.d1_camalign_scores) if args.d1_camalign_scores else None,
    )
    write_csv(args.out_csv, table)
    write_md(args.out_md, table, summary)
    write_json(args.out_summary, summary)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_md}")
    print(f"Wrote {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
