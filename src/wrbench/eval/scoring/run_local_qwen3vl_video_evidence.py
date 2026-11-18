#!/usr/bin/env python3
"""Local Qwen3-VL structured subject judgeability evidence runner.

Runs either:
- ``subject_judgeability_v1``: binary per-second subject visibility/judgeability scan.
- ``guarded_teacher_gate_v3``: guarded N/A rescue pass distinguishing missing
  returned evidence from wrong-but-judgeable outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from wrbench.eval._io import (
    read_json,
    read_records,
    select_manifest_shard,
    sha256_file,
    stable_shard,
    write_json,
    write_jsonl as _io_write_jsonl,
)

try:
    from .runtime_common import (
        cuda_build_or_unavailable as _cuda_build_or_unavailable,
        derive_frames_used_from_processor_inputs,
        explicit_cuda_placement_metadata,
        index_by_video_id,
        load_model_explicit_cuda,
        model_config_model_type,
        scoring_video_path,
        version_or_unavailable as _version_or_unavailable,
    )
except ImportError:
    from wrbench.eval.scoring.runtime_common import (
        cuda_build_or_unavailable as _cuda_build_or_unavailable,
        derive_frames_used_from_processor_inputs,
        explicit_cuda_placement_metadata,
        index_by_video_id,
        load_model_explicit_cuda,
        model_config_model_type,
        scoring_video_path,
        version_or_unavailable as _version_or_unavailable,
    )


SUBJECT_SCHEMA_VERSION = "qwen3vl_subject_judgeability_v1"
GUARDED_SCHEMA_VERSION = "qwen3vl_guarded_teacher_gate_v3"
SCHEMA_VERSION = SUBJECT_SCHEMA_VERSION
PROMPT_SCHEMA_SUBJECT = "subject_judgeability_v1"
PROMPT_SCHEMA_GUARDED = "guarded_teacher_gate_v3"
SUPPORTED_PROMPT_SCHEMAS = {PROMPT_SCHEMA_SUBJECT, PROMPT_SCHEMA_GUARDED}

DEFAULT_MODEL_PATH = ""
DEFAULT_FPS = "4"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_ATTN_IMPLEMENTATION = "flash_attention_2"
DEFAULT_MAX_NEW_TOKENS = 768
RAW_FILENAME = "raw_qwen3vl_judgeability_evidence.jsonl"
PARSED_FILENAME = "parsed_qwen3vl_judgeability_evidence.jsonl"
EVIDENCE_FILENAME = "evidence.jsonl"
JUDGEABILITY_VALUES = {"yes", "no"}
REQUIRED_KEYS = ["video_id", "main_subject", "per_second"]

load_json = read_json


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_records(path, allow_missing=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _io_write_jsonl(rows, path)


def id_map(rows: list[dict[str, Any]], *, label: str = "rows") -> dict[str, dict[str, Any]]:
    return index_by_video_id(rows, label=label, require_unique=True)


PROMPT_TEMPLATE = """You are a strict video judgeability auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the single main prompt-critical subject.
2. Inspect the video in temporal order, one whole second at a time.
3. For each second, decide whether the same main subject is judgeable.

judgeable = yes only when:
- the same main subject is visible and identifiable enough;
- its coarse position can be judged;
- its action or state can be judged.

judgeable = no when:
- the subject is out of frame, fully occluded, too blurred, too small, or unidentifiable;
- or only a fragment is visible and coarse position/action/state cannot be judged.

Do not score D5 or D6. Do not decide whether the video is correct. Do not explain beyond short visual notes.

Text prompt:
{world_state_prompt}

Video id:
{video_id}

Return this exact JSON schema:
{{
  "video_id": "<string>",
  "main_subject": "<short subject name>",
  "per_second": [
    {{
      "sec": <integer second>,
      "judgeable": "<yes|no>",
      "note": "<short visual note>"
    }}
  ]
}}"""


GUARDED_PROMPT_TEMPLATE = """You are a strict D5/D6 judgeability gate auditor.

Return only valid JSON. Do not return markdown.

Task:
1. Identify the prompt-critical subject, object, target, result evidence, or region needed to judge the prompt.
2. Identify static anchors that help compare before/after positions or state.
3. Inspect the video in temporal order, one whole second at a time.
4. Decide only whether D5/D6 are judgeable/applicable. Do not decide whether the video is correct.

Critical gate rules:
- Wrong but visible and identifiable returned evidence is applicable.
- A visible but irrelevant object is not enough; evidence must be prompt-critical.
- If the object/person/state returns in a wrong place or wrong state but can be visually judged, D5/D6 are applicable.
- Mark N/A only for no usable returned evidence, no OoV interval, no return after OoV, unidentifiable evidence, or completely unjudgeable evidence.
- Do not use N/A for incorrect-but-visible outcomes.

Text prompt:
{world_state_prompt}

Video id:
{video_id}

Return this exact JSON schema:
{{
  "video_id": "<string>",
  "prompt_critical_evidence": "<short subject/object/target/result name>",
  "expected_motion_or_state": "<short phrase>",
  "static_anchors": ["<short anchor>", "..."],
  "before_hidden_evidence": "<short|null>",
  "after_return_evidence": "<short|null>",
  "per_second": [
    {{
      "sec": <integer second>,
      "judgeable": "<yes|no>",
      "note": "<short prompt-critical visual note>"
    }}
  ],
  "gate": {{
    "d5_applicable": <true|false>,
    "d6_applicable": <true|false>,
    "na_reason": "<no_usable_returned_evidence|no_oov|no_return|unidentifiable|completely_unjudgeable|unclear|null>"
  }},
  "audit": {{
    "wrong_but_judgeable": <true|false>,
    "anchor_available": <true|false>,
    "after_return_available": <true|false>
  }}
}}"""


def attach_outer_request_metadata(manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row_index, item in enumerate(manifest):
        item_copy = dict(item)
        video_id = str(item_copy.get("video_id") or "")
        item_copy["row_index"] = int(item_copy.get("row_index", row_index))
        if not item_copy.get("request_id") and video_id:
            digest = hashlib.sha256(f"{row_index}:{video_id}".encode("utf-8")).hexdigest()[:16]
            item_copy["request_id"] = f"qwen3vl-{row_index:06d}-{digest}"
        annotated.append(item_copy)
    return annotated


def build_judgeability_prompt(
    *,
    world_state_prompt: str,
    video_id: str,
    prompt_schema: str = PROMPT_SCHEMA_SUBJECT,
) -> str:
    if prompt_schema == PROMPT_SCHEMA_GUARDED:
        return GUARDED_PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt, video_id=video_id)
    if prompt_schema == PROMPT_SCHEMA_SUBJECT:
        return PROMPT_TEMPLATE.format(world_state_prompt=world_state_prompt, video_id=video_id)
    raise ValueError(f"unsupported prompt_schema: {prompt_schema}")


def schema_version_for_prompt_schema(prompt_schema: str) -> str:
    if prompt_schema == PROMPT_SCHEMA_GUARDED:
        return GUARDED_SCHEMA_VERSION
    if prompt_schema == PROMPT_SCHEMA_SUBJECT:
        return SUBJECT_SCHEMA_VERSION
    raise ValueError(f"unsupported prompt_schema: {prompt_schema}")


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first valid JSON object from a model response."""
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    candidate = match.group(1).strip() if match else stripped
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed
    decoder = json.JSONDecoder()
    for found in re.finditer(r"\{", text):
        try:
            parsed, _ = decoder.raw_decode(text[found.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no valid JSON object found in model output")


def _require_type(errors: list[str], payload: dict[str, Any], key: str, expected_type: type | tuple[type, ...]) -> None:
    if key not in payload:
        errors.append(f"missing key: {key}")
    elif not isinstance(payload[key], expected_type):
        errors.append(f"{key} must be {expected_type}")


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped and stripped.lower() != "null" else None
    return str(value).strip() or None


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
    return None


def _bool_or_error(errors: list[str], value: Any, key: str) -> bool:
    parsed = _optional_bool(value)
    if parsed is None:
        errors.append(f"{key} must be boolean")
        return False
    return parsed


def _normalize_per_second(
    errors: list[str],
    entries: Any,
    *,
    note_max_len: int,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries or []):
        if not isinstance(entry, dict):
            errors.append(f"per_second[{index}] must be object")
            continue
        for key in ("sec", "judgeable", "note"):
            if key not in entry:
                errors.append(f"per_second[{index}] missing {key}")
        try:
            sec = int(entry.get("sec"))
        except (TypeError, ValueError):
            errors.append(f"per_second[{index}].sec must be integer")
            sec = index
        judgeable = str(entry.get("judgeable") or "").strip()
        if judgeable not in JUDGEABILITY_VALUES:
            errors.append(f"per_second[{index}].judgeable invalid: {entry.get('judgeable')!r}")
        note = entry.get("note")
        if not isinstance(note, str):
            errors.append(f"per_second[{index}].note must be string")
            note = "" if note is None else str(note)
        normalized.append({"sec": sec, "judgeable": judgeable, "note": str(note).strip()[:note_max_len]})
    return sorted(normalized, key=lambda row: row["sec"])


def _check_video_id(
    errors: list[str],
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    allow_video_id_repair: bool,
) -> dict[str, Any]:
    if payload.get("video_id") != expected_video_id and allow_video_id_repair and isinstance(payload.get("video_id"), str):
        payload = dict(payload)
        payload["video_id"] = expected_video_id
    if payload.get("video_id") != expected_video_id:
        errors.append(f"video_id mismatch: expected {expected_video_id!r}, got {payload.get('video_id')!r}")
    return payload


def _derive_temporal_oov_gate(
    normalized_rows: list[dict[str, Any]],
    *,
    judgeable_key: str = "judgeable",
) -> dict[str, Any]:
    """Derive the hard D5/D6 applicability gate from sampled visibility rows."""
    first_gap_index: int | None = None
    saw_prior_judgeable = False
    for index, entry in enumerate(normalized_rows):
        judgeable = str(entry.get(judgeable_key) or "")
        if judgeable == "yes":
            saw_prior_judgeable = True
        elif judgeable == "no" and saw_prior_judgeable:
            first_gap_index = index
            break

    if first_gap_index is not None:
        gap_row = normalized_rows[first_gap_index]
        start_sec = float(gap_row.get("sec", first_gap_index))
        end_sec = start_sec + 1.0
        for entry in normalized_rows[first_gap_index + 1 :]:
            if str(entry.get(judgeable_key) or "") == "no":
                end_sec = float(entry.get("sec", end_sec)) + 1.0
            else:
                break
        oov_interval = {"start_sec": start_sec, "end_sec": end_sec, "status": "present"}
        return_judgeable = any(str(entry.get(judgeable_key) or "") == "yes" for entry in normalized_rows[first_gap_index + 1 :])
        na_reason = None if return_judgeable else "no_return"
    else:
        oov_interval = {"start_sec": None, "end_sec": None, "status": "absent"}
        return_judgeable = False
        na_reason = "no_initial_judgeable" if any(str(entry.get(judgeable_key) or "") == "no" for entry in normalized_rows) else "no_oov"

    return {
        "first_gap_index": first_gap_index,
        "oov_interval": oov_interval,
        "return_judgeable": bool(return_judgeable),
        "applicable": na_reason is None,
        "na_reason": na_reason,
    }


def add_evidence_first_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    timeline = parsed.get("per_second") or []
    segments = [
        {
            "start_sec": float(entry.get("sec", index)),
            "end_sec": float(entry.get("sec", index)) + 1.0,
            "visible_subject": str(entry.get("judgeable") or "no").strip(),
            "note": str(entry.get("note") or "").strip(),
        }
        for index, entry in enumerate(timeline)
        if isinstance(entry, dict)
    ]
    temporal_gate = _derive_temporal_oov_gate(timeline, judgeable_key="judgeable")
    first_gap_index = temporal_gate["first_gap_index"]
    oov_interval = temporal_gate["oov_interval"]
    return_judgeable = bool(temporal_gate["return_judgeable"])
    na_reason = temporal_gate["na_reason"]
    evidence_applicable = bool(temporal_gate["applicable"])
    judgeable_flag = bool(return_judgeable or first_gap_index is None)
    parsed.update(
        {
            "segments": segments,
            "subject_judgeable": judgeable_flag,
            "target_judgeable": judgeable_flag,
            "action_state_judgeable": judgeable_flag,
            "oov_interval": oov_interval,
            "return_judgeable": return_judgeable,
            "notes_short": "; ".join(str(entry.get("note") or "").strip() for entry in timeline[:3] if str(entry.get("note") or "").strip()),
            "shared_oov_applicable": evidence_applicable,
            "shared_oov_na_reason": na_reason,
            "evidence_shared_oov_applicable": evidence_applicable,
            "evidence_shared_oov_na_reason": na_reason,
            "evidence_subject_judgeable": judgeable_flag,
            "evidence_target_judgeable": judgeable_flag,
            "evidence_action_state_judgeable": judgeable_flag,
            "evidence_oov_interval_status": oov_interval["status"],
            "evidence_return_judgeable": return_judgeable,
            "evidence_d5_applicable": evidence_applicable,
            "evidence_d6_applicable": evidence_applicable,
            "evidence_d5_na_reason": None if evidence_applicable else na_reason or "unclear",
            "evidence_d6_na_reason": None if evidence_applicable else na_reason or "unclear",
        }
    )
    return parsed


def add_guarded_teacher_gate_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    timeline = parsed.get("per_second") or []
    normalized_rows = [
        {
            "sec": int(entry.get("sec", index)),
            "judgeable": str(entry.get("judgeable") or "no").strip(),
            "note": str(entry.get("note") or "").strip(),
        }
        for index, entry in enumerate(timeline)
        if isinstance(entry, dict)
    ]
    segments = [
        {
            "start_sec": float(row["sec"]),
            "end_sec": float(row["sec"]) + 1.0,
            "visible_subject": row["judgeable"],
            "prompt_critical_judgeable": row["judgeable"],
            "note": row["note"],
        }
        for row in normalized_rows
    ]
    temporal_gate = _derive_temporal_oov_gate(normalized_rows, judgeable_key="judgeable")
    oov_interval = temporal_gate["oov_interval"]
    return_judgeable = bool(temporal_gate["return_judgeable"])
    gate = parsed.get("gate") or {}
    model_d5_app = bool(gate.get("d5_applicable"))
    model_d6_app = bool(gate.get("d6_applicable"))
    model_na_reason = _string_or_none(gate.get("na_reason"))
    if not temporal_gate["applicable"]:
        d5_app = d6_app = shared_app = False
        shared_reason: str | None = str(temporal_gate["na_reason"] or "unclear")
    else:
        d5_app, d6_app = model_d5_app, model_d6_app
        shared_app = bool(d5_app and d6_app)
        shared_reason = None if shared_app else model_na_reason or "unclear"
    audit = parsed.get("audit") if isinstance(parsed.get("audit"), dict) else {}
    anchor_available = _optional_bool(audit.get("anchor_available"))
    target_judgeable = bool(anchor_available if anchor_available is not None else return_judgeable)
    parsed.update(
        {
            "per_second": normalized_rows,
            "main_subject": parsed.get("prompt_critical_evidence"),
            "segments": segments,
            "subject_judgeable": return_judgeable,
            "target_judgeable": target_judgeable,
            "action_state_judgeable": return_judgeable,
            "oov_interval": oov_interval,
            "return_judgeable": return_judgeable,
            "notes_short": "; ".join(row["note"] for row in normalized_rows[:3] if row["note"]),
            "shared_oov_applicable": shared_app,
            "shared_oov_na_reason": shared_reason,
            "evidence_shared_oov_applicable": shared_app,
            "evidence_shared_oov_na_reason": shared_reason,
            "evidence_subject_judgeable": return_judgeable,
            "evidence_target_judgeable": target_judgeable,
            "evidence_action_state_judgeable": return_judgeable,
            "evidence_oov_interval_status": oov_interval["status"],
            "evidence_return_judgeable": return_judgeable,
            "evidence_d5_applicable": d5_app,
            "evidence_d6_applicable": d6_app,
            "evidence_d5_na_reason": None if d5_app else shared_reason,
            "evidence_d6_na_reason": None if d6_app else shared_reason,
        }
    )
    return parsed


def validate_judgeability_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    for key in REQUIRED_KEYS:
        if key not in payload:
            errors.append(f"missing key: {key}")
    payload = _check_video_id(
        errors, payload, expected_video_id=expected_video_id, allow_video_id_repair=allow_video_id_repair
    )
    _require_type(errors, payload, "main_subject", str)
    _require_type(errors, payload, "per_second", list)
    normalized_per_second = _normalize_per_second(errors, payload.get("per_second"), note_max_len=160)
    if errors:
        raise ValueError("; ".join(errors))
    return add_evidence_first_fields(
        {
            "video_id": payload.get("video_id"),
            "main_subject": str(payload.get("main_subject") or "").strip(),
            "per_second": normalized_per_second,
            "schema_version": SCHEMA_VERSION,
        }
    )


def validate_guarded_teacher_gate_payload(
    payload: dict[str, Any],
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    required_keys = [
        "video_id",
        "prompt_critical_evidence",
        "expected_motion_or_state",
        "static_anchors",
        "before_hidden_evidence",
        "after_return_evidence",
        "per_second",
        "gate",
    ]
    for key in required_keys:
        if key not in payload:
            errors.append(f"missing key: {key}")
    payload = _check_video_id(
        errors, payload, expected_video_id=expected_video_id, allow_video_id_repair=allow_video_id_repair
    )
    _require_type(errors, payload, "prompt_critical_evidence", str)
    _require_type(errors, payload, "expected_motion_or_state", str)
    _require_type(errors, payload, "static_anchors", list)
    _require_type(errors, payload, "per_second", list)
    _require_type(errors, payload, "gate", dict)

    anchors: list[str] = []
    if isinstance(payload.get("static_anchors"), list):
        for index, value in enumerate(payload["static_anchors"]):
            if not isinstance(value, str):
                errors.append(f"static_anchors[{index}] must be string")
            elif value.strip():
                anchors.append(value.strip()[:120])

    normalized_per_second = _normalize_per_second(errors, payload.get("per_second"), note_max_len=180)
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    d5_app = _bool_or_error(errors, gate.get("d5_applicable"), "gate.d5_applicable")
    d6_app = _bool_or_error(errors, gate.get("d6_applicable"), "gate.d6_applicable")
    na_reason = _string_or_none(gate.get("na_reason"))
    allowed_na_reasons = {
        "no_usable_returned_evidence",
        "no_oov",
        "no_return",
        "no_initial_judgeable",
        "unidentifiable",
        "unjudgeable_return",
        "completely_unjudgeable",
        "parse_error",
        "unclear",
    }
    if (not d5_app or not d6_app) and na_reason not in allowed_na_reasons:
        errors.append(f"gate.na_reason invalid: {gate.get('na_reason')!r}")

    audit_payload = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
    audit = {
        key: _optional_bool(audit_payload.get(key))
        for key in ("wrong_but_judgeable", "anchor_available", "after_return_available")
    }
    if errors:
        raise ValueError("; ".join(errors))

    return add_guarded_teacher_gate_fields(
        {
            "video_id": payload.get("video_id"),
            "prompt_critical_evidence": str(payload.get("prompt_critical_evidence") or "").strip(),
            "expected_motion_or_state": str(payload.get("expected_motion_or_state") or "").strip()[:240],
            "static_anchors": anchors,
            "before_hidden_evidence": _string_or_none(payload.get("before_hidden_evidence")),
            "after_return_evidence": _string_or_none(payload.get("after_return_evidence")),
            "per_second": normalized_per_second,
            "gate": {"d5_applicable": d5_app, "d6_applicable": d6_app, "na_reason": None if d5_app and d6_app else na_reason},
            "audit": audit,
            "schema_version": GUARDED_SCHEMA_VERSION,
        }
    )


def parse_judgeability_response(
    raw_text: str,
    *,
    expected_video_id: str,
    allow_video_id_repair: bool = False,
    prompt_schema: str = PROMPT_SCHEMA_SUBJECT,
) -> dict[str, Any]:
    payload = extract_json_object(raw_text)
    if prompt_schema == PROMPT_SCHEMA_GUARDED:
        return validate_guarded_teacher_gate_payload(
            payload, expected_video_id=expected_video_id, allow_video_id_repair=allow_video_id_repair
        )
    if prompt_schema == PROMPT_SCHEMA_SUBJECT:
        return validate_judgeability_payload(
            payload, expected_video_id=expected_video_id, allow_video_id_repair=allow_video_id_repair
        )
    raise ValueError(f"unsupported prompt_schema: {prompt_schema}")


def validate_visibility_payload(payload: dict[str, Any], *, expected_video_id: str) -> dict[str, Any]:
    return validate_judgeability_payload(payload, expected_video_id=expected_video_id)


def parse_visibility_response(raw_text: str, *, expected_video_id: str) -> dict[str, Any]:
    return parse_judgeability_response(raw_text, expected_video_id=expected_video_id)


class LocalQwen3VLVideoEvidenceRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        fps: str,
        dtype: str,
        attn_implementation: str,
        local_rank: int,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        prompt_schema: str = PROMPT_SCHEMA_SUBJECT,
    ) -> None:
        if prompt_schema not in SUPPORTED_PROMPT_SCHEMAS:
            raise ValueError(f"unsupported prompt_schema: {prompt_schema}")
        self.model_path = model_path
        self.fps = fps
        self.max_new_tokens = int(max_new_tokens)
        self.prompt_schema = prompt_schema
        import torch  # type: ignore
        from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore

        dtype_obj = {"bfloat16": torch.bfloat16}[dtype]
        self.processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        video_backend = os.environ["WORLD_STATE_VIDEO_BACKEND"].strip()
        if not video_backend:
            raise RuntimeError("WORLD_STATE_VIDEO_BACKEND is required")
        video_processor = getattr(self.processor, "video_processor", None)
        if video_backend and video_processor is not None:
            from transformers.video_utils import load_video  # type: ignore

            def fetch_videos_with_backend(video_url_or_urls, sample_indices_fn=None):
                if isinstance(video_url_or_urls, list):
                    return list(zip(*[fetch_videos_with_backend(x, sample_indices_fn=sample_indices_fn) for x in video_url_or_urls]))
                return load_video(video_url_or_urls, backend=video_backend, sample_indices_fn=sample_indices_fn)

            video_processor.fetch_videos = fetch_videos_with_backend

        def load_with(model_cls: Any) -> Any:
            return load_model_explicit_cuda(
                model_cls,
                model_path=str(model_path),
                dtype=dtype_obj,
                attn_implementation=attn_implementation,
                local_rank=local_rank,
            )

        try:
            from transformers import Qwen3VLForConditionalGeneration  # type: ignore

            self.model = load_with(Qwen3VLForConditionalGeneration)
        except Exception:
            self.model = load_with(AutoModelForImageTextToText)
        self.model.eval()

    def run_one(self, item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        import torch  # type: ignore

        video_id = str(item["video_id"])
        request_id = str(item.get("request_id") or "")
        row_index = item.get("row_index")
        video_path = scoring_video_path(item)
        image_path = str(item.get("image_path") or item.get("sheet_path") or "")
        media_type = "image" if image_path else "video"
        media_path = image_path or video_path
        world_state_prompt = str(item.get("world_state_prompt") or item.get("prompt_text") or "")
        prompt = build_judgeability_prompt(
            world_state_prompt=world_state_prompt, video_id=video_id, prompt_schema=self.prompt_schema
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": media_type, media_type: media_path},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        processor_kwargs: dict[str, Any] = {}
        if media_type == "video" and self.fps != "full":
            processor_kwargs["videos_kwargs"] = {"fps": float(self.fps)}
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs=processor_kwargs or None,
        )
        frames_used = derive_frames_used_from_processor_inputs(inputs, self.processor)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, temperature=None, top_p=None
            )
        prompt_len = inputs["input_ids"].shape[-1]
        raw_text = self.processor.tokenizer.decode(output_ids[0][prompt_len:], skip_special_tokens=True).strip()
        raw_row: dict[str, Any] = {
            "video_id": video_id,
            "request_id": request_id,
            "row_index": row_index,
            "path": media_path,
            "video_path": video_path,
            "image_path": image_path,
            "media_type": media_type,
            "world_state_prompt": world_state_prompt,
            "sampling_fps": self.fps,
            "frames_used": int(frames_used),
            "raw_response": raw_text,
            "schema_version": schema_version_for_prompt_schema(self.prompt_schema),
            "prompt_schema": self.prompt_schema,
        }
        try:
            parsed_row = parse_judgeability_response(
                raw_text, expected_video_id=video_id, allow_video_id_repair=True, prompt_schema=self.prompt_schema
            )
            parsed_row.update(
                {
                    "request_id": request_id,
                    "row_index": row_index,
                    "path": media_path,
                    "video_path": video_path,
                    "image_path": image_path,
                    "media_type": media_type,
                    "world_state_prompt": world_state_prompt,
                    "sampling_fps": self.fps,
                    "frames_used": int(frames_used),
                }
            )
            raw_row["parse_status"] = "ok"
            return raw_row, parsed_row
        except Exception as exc:
            raw_row["parse_status"] = "error"
            raw_row["parse_error"] = f"{type(exc).__name__}: {exc}"
            return raw_row, None


def merge_sharded_outputs(
    *,
    output_dir: Path,
    manifest: list[dict[str, Any]],
    require_complete: bool = True,
) -> dict[str, Any]:
    raw_chunk_paths = sorted(output_dir.glob("raw_qwen3vl_judgeability_evidence_shard_*.jsonl"))
    parsed_chunk_paths = sorted(output_dir.glob("parsed_qwen3vl_judgeability_evidence_shard_*.jsonl"))
    if not raw_chunk_paths and (output_dir / RAW_FILENAME).exists():
        raw_chunk_paths = [output_dir / RAW_FILENAME]
    if not parsed_chunk_paths and (output_dir / PARSED_FILENAME).exists():
        parsed_chunk_paths = [output_dir / PARSED_FILENAME]
    raw_rows = [row for path in raw_chunk_paths for row in load_jsonl(path)]
    parsed_rows = [row for path in parsed_chunk_paths for row in load_jsonl(path)]
    raw_by_id = id_map(raw_rows, label="raw evidence rows")
    parsed_by_id = id_map(parsed_rows, label="parsed evidence rows")
    manifest_ids = [str(item.get("video_id")) for item in manifest if item.get("video_id")]
    manifest_id_set = set(manifest_ids)
    missing_raw = [vid for vid in manifest_ids if vid not in raw_by_id]
    missing_parsed = [vid for vid in manifest_ids if vid not in parsed_by_id]
    if require_complete and missing_raw:
        raise RuntimeError(f"cannot merge incomplete evidence output: {len(missing_raw)} manifest videos are missing")
    ordered_raw = [raw_by_id[vid] for vid in manifest_ids if vid in raw_by_id]
    ordered_parsed = [parsed_by_id[vid] for vid in manifest_ids if vid in parsed_by_id]
    write_jsonl(output_dir / RAW_FILENAME, ordered_raw)
    write_jsonl(output_dir / PARSED_FILENAME, ordered_parsed)
    write_jsonl(output_dir / EVIDENCE_FILENAME, ordered_parsed)
    summary = {
        "schema_version": str(ordered_parsed[0].get("schema_version")) if ordered_parsed and ordered_parsed[0].get("schema_version") else SCHEMA_VERSION,
        "merge_status": "complete" if not missing_raw else "incomplete",
        "manifest_records": len(manifest_ids),
        "raw_records_written": len(ordered_raw),
        "parsed_records_written": len(ordered_parsed),
        "evidence_records_written": len(ordered_parsed),
        "parse_rate": round(len(ordered_parsed) / len(ordered_raw), 6) if ordered_raw else 0.0,
        "missing_video_ids": missing_raw,
        "missing_parsed_video_ids": missing_parsed,
        "unused_raw_video_ids": sorted(vid for vid in raw_by_id if vid not in manifest_id_set),
        "unused_parsed_video_ids": sorted(vid for vid in parsed_by_id if vid not in manifest_id_set),
        "request_mapping": [
            {
                "video_id": str(item.get("video_id")),
                "request_id": item.get("request_id"),
                "row_index": item.get("row_index"),
                "raw_present": str(item.get("video_id")) in raw_by_id,
                "parsed_present": str(item.get("video_id")) in parsed_by_id,
            }
            for item in manifest
            if item.get("video_id")
        ],
        "raw_chunk_files": [str(path) for path in raw_chunk_paths],
        "parsed_chunk_files": [str(path) for path in parsed_chunk_paths],
    }
    write_json(output_dir / "merge_summary_qwen3vl_judgeability_evidence.json", summary)
    return summary


def build_run_config(
    *,
    args: argparse.Namespace,
    raw_records_written: int,
    parsed_records_written: int,
    records_expected: int,
    max_frames_observed: int,
    local_rank: int,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version_for_prompt_schema(args.prompt_schema),
        "prompt_schema": args.prompt_schema,
        "fps": str(args.fps),
        "max_frames_observed": int(max_frames_observed),
        "model_path": str(args.model_path),
        "model_config_model_type": model_config_model_type(args.model_path),
        "python_path": sys.executable,
        "transformers_version": _version_or_unavailable("transformers"),
        "torch_version": _version_or_unavailable("torch"),
        "cuda_build": _cuda_build_or_unavailable(),
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": int(args.max_new_tokens),
        "video_sampling_policy": "processor_default_not_all_source_frames" if str(args.fps) == "full" else "processor_kwargs_fps",
        **explicit_cuda_placement_metadata(local_rank),
        "local_rank": int(local_rank),
        "num_shards": int(args.num_shards),
        "shard_id": int(args.shard_id),
        "input_manifest_path": str(args.manifest_path),
        "input_manifest_sha256": sha256_file(args.manifest_path),
        "records_expected": int(records_expected),
        "raw_records_written": int(raw_records_written),
        "parsed_records_written": int(parsed_records_written),
        "parse_rate": round(parsed_records_written / raw_records_written, 6) if raw_records_written else 0.0,
        "skip_existing": bool(args.skip_existing),
    }


def stage_result_text(*, config: dict[str, Any]) -> str:
    parse_rate = float(config.get("parse_rate") or 0.0)
    records_expected = int(config.get("records_expected") or 0)
    raw_records_written = int(config.get("raw_records_written") or 0)
    decision = "pass" if raw_records_written >= records_expected and parse_rate >= 0.95 else "retry_once"
    prompt_schema = str(config.get("prompt_schema") or PROMPT_SCHEMA_SUBJECT)
    return (
        f"# Stage {config.get('schema_version') or SCHEMA_VERSION}\n\n"
        f"- scorer: qwen3vl_video_evidence/{prompt_schema}\n"
        f"- decision: {decision}\n\n"
        "## metrics\n"
        f"- records_expected: {records_expected}\n"
        f"- raw_records_written: {raw_records_written}\n"
        f"- parsed_records_written: {config.get('parsed_records_written')}\n"
        f"- parse_rate: {parse_rate:.6f}\n"
        f"- max_frames_observed: {config.get('max_frames_observed')}\n\n"
        "## notes\n"
        "- Candidate subject judgeability evidence generated.\n"
        "- Human N/A diagnostic mapping is post-processed from yes -> no -> yes transitions.\n"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Qwen3-VL structured subject judgeability evidence runner")
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Local Qwen3-VL model directory (set via wrbench.runtime.json eval.scorers.qwen3vl_model).")
    parser.add_argument("--fps", required=True)
    parser.add_argument("--max-videos", type=int, required=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--allow-incomplete-merge", action="store_true")
    parser.add_argument("--dtype", required=True)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--local-rank", type=int, required=True)
    parser.add_argument("--prompt-schema", choices=sorted(SUPPORTED_PROMPT_SCHEMAS), required=True)
    args = parser.parse_args(argv)
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard-id must satisfy 0 <= shard_id < num_shards")
    if args.max_videos < 0:
        parser.error("--max-videos must be >= 0")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.dtype != DEFAULT_DTYPE:
        parser.error("--dtype must be bfloat16 for this route")
    if args.attn_implementation != DEFAULT_ATTN_IMPLEMENTATION:
        parser.error("--attn-implementation must be flash_attention_2 for this route")
    if str(args.fps) != "full":
        try:
            fps_value = float(args.fps)
        except ValueError:
            parser.error("--fps must be a positive number or full")
        if fps_value <= 0:
            parser.error("--fps must be positive or full")
        args.fps = str(int(fps_value)) if fps_value.is_integer() else str(fps_value)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    local_rank = int(args.local_rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_json(args.manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("--manifest-path must contain a JSON list")
    manifest = attach_outer_request_metadata(manifest)

    selected_manifest = manifest[: args.max_videos] if args.max_videos else manifest
    if args.merge_only:
        summary = merge_sharded_outputs(
            output_dir=args.output_dir,
            manifest=selected_manifest,
            require_complete=not args.allow_incomplete_merge,
        )
        config = build_run_config(
            args=args,
            raw_records_written=int(summary["raw_records_written"]),
            parsed_records_written=int(summary["parsed_records_written"]),
            records_expected=int(summary["manifest_records"]),
            max_frames_observed=0,
            local_rank=local_rank,
        )
        write_json(args.output_dir / "run_config.json", config)
        (args.output_dir / "stage_result.md").write_text(stage_result_text(config=config), encoding="utf-8")
        return 0

    shard_items = select_manifest_shard(manifest, num_shards=args.num_shards, shard_id=args.shard_id)
    if args.max_videos:
        shard_items = shard_items[: args.max_videos]

    raw_path = args.output_dir / (RAW_FILENAME if args.num_shards == 1 else f"raw_qwen3vl_judgeability_evidence_shard_{args.shard_id}.jsonl")
    parsed_path = args.output_dir / (PARSED_FILENAME if args.num_shards == 1 else f"parsed_qwen3vl_judgeability_evidence_shard_{args.shard_id}.jsonl")
    existing_raw = load_jsonl(raw_path) if args.skip_existing else []
    existing_parsed = load_jsonl(parsed_path) if args.skip_existing else []
    existing_raw_by_id = id_map(existing_raw, label="existing raw rows")

    runner = LocalQwen3VLVideoEvidenceRunner(
        model_path=args.model_path,
        fps=str(args.fps),
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        local_rank=local_rank,
        max_new_tokens=args.max_new_tokens,
        prompt_schema=args.prompt_schema,
    )
    new_raw: list[dict[str, Any]] = []
    new_parsed: list[dict[str, Any]] = []
    for item in shard_items:
        if args.skip_existing and str(item["video_id"]) in existing_raw_by_id:
            continue
        raw_row, parsed_row = runner.run_one(item)
        new_raw.append(raw_row)
        if parsed_row is not None:
            new_parsed.append(parsed_row)

    raw_by_id = id_map(existing_raw + new_raw, label="raw rows")
    parsed_by_id = id_map(existing_parsed + new_parsed, label="parsed rows")
    ordered_raw = [raw_by_id[str(item["video_id"])] for item in shard_items if str(item.get("video_id")) in raw_by_id]
    ordered_parsed = [parsed_by_id[str(item["video_id"])] for item in shard_items if str(item.get("video_id")) in parsed_by_id]
    write_jsonl(raw_path, ordered_raw)
    write_jsonl(parsed_path, ordered_parsed)
    if args.num_shards == 1:
        write_jsonl(args.output_dir / EVIDENCE_FILENAME, ordered_parsed)

    config = build_run_config(
        args=args,
        raw_records_written=len(ordered_raw),
        parsed_records_written=len(ordered_parsed),
        records_expected=len(shard_items),
        max_frames_observed=max([int(row.get("frames_used") or 0) for row in ordered_raw] or [0]),
        local_rank=local_rank,
    )
    write_json(args.output_dir / ("run_config.json" if args.num_shards == 1 else f"run_config_shard_{args.shard_id}.json"), config)
    (args.output_dir / ("stage_result.md" if args.num_shards == 1 else f"stage_result_shard_{args.shard_id}.md")).write_text(
        stage_result_text(config=config),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
