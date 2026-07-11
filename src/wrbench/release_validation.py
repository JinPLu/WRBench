"""Validation for immutable WRBench paper-release provenance bundles."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from wrbench.datasets import (
    NATURAL25_PAPER_RELEASE_ID,
    NATURAL25_RELEASE_CORE_FILES,
    load_jsonl,
    natural25_release_dir,
    natural25_variants_path,
)


LOCAL_PROMPT_SHA256 = "19d99d90b40fe3453d74a634d615403fe4721597422f1d9d433df1611901ac7e"
API_SOURCE_PROMPT_SHA256 = "ce4eea268ee0c0f7fb379be8a51eb9920e66cee1e18ad82dd7111eb9c99a2ad7"
TOOLKIT_VARIANTS_SHA256 = "35d0fe92aa685afb1d32ca26284b1ab3b6a98d7ef692696831392ab24b3c8b34"
VIDEO_ASSET_REVISION = "8a927b9322c5d8af6474399ce4840ef4148f8e39"
EVENT_TIERS = {"T0", "T1", "T2_div_a", "T2_div_b"}
_HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")

LOCAL_500_MODELS = (
    "hunyuan_worldplay",
    "lingbot_world",
    "lingbot_world_act",
    "liveworld",
    "magicworld",
    "spatia",
    "versecrafter",
    "wan21_fun_14b_cam",
    "wan21_fun_1p3b_cam",
    "wan22_fun_5b_cam",
    "wan22_fun_a14b_cam",
)
LOCAL_400_MODELS = (
    "gen3c",
    "hunyuan_game_craft",
    "hydra",
    "inspatio_world_14b",
    "recammaster",
)
API_300_MODELS = (
    "hailuo_2_3",
    "happyhorse_1_0_i2v",
    "kling_v2_6",
    "wan2_2_i2v_plus",
    "wan2_6_i2v",
    "wan2_7_i2v",
    "wanx2_1_i2v_turbo",
)
EXPECTED_MODEL_ROWS = {
    **{model: 500 for model in LOCAL_500_MODELS},
    **{model: 400 for model in LOCAL_400_MODELS},
    **{model: 300 for model in API_300_MODELS},
}
EXPECTED_MODEL_CATALOG = {
    **{model: "local_ti2v_tv2v" for model in (*LOCAL_500_MODELS, *LOCAL_400_MODELS)},
    **{model: "api_source" for model in API_300_MODELS},
}
EXPECTED_LOCAL_DUAL_CELLS = {
    ("yaw_LR", 30),
    ("yaw_RL", 30),
    ("yaw_LR", 60),
    ("yaw_RL", 60),
}
EXPECTED_API_CELLS = {
    ("static", None),
    ("yaw_LR", None),
    ("yaw_RL", None),
}
EXPECTED_LOCAL_CURRENT_STATIC = {
    **{model: True for model in LOCAL_500_MODELS},
    "gen3c": False,
    "hunyuan_game_craft": True,
    "hydra": False,
    "inspatio_world_14b": False,
    "recammaster": False,
}
EXPECTED_ARTIFACT_ROWS = {
    "README.md": 1,
    "variants.local_ti2v_tv2v.jsonl": 400,
    "variants.api_source.jsonl": 400,
    "prompt_usage.json": 23,
    "camera_scope.json": 23,
    "camera_scopes/local_dual_angle.json": 64,
    "camera_scopes/local_static.json": 11,
    "camera_scopes/api_prompt_camera.json": 21,
    "tv2v_sources.jsonl": 100,
    "hydra_evaluation_policy.json": 1,
}


class ReleaseValidationError(ValueError):
    """Raised when a bundled release violates the public contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseValidationError(f"{path}: expected a JSON object")
    return payload


def _event_tier(variant_id: str) -> str:
    parts = variant_id.split("__")
    if len(parts) < 3 or parts[-2] not in EVENT_TIERS:
        raise ReleaseValidationError(f"invalid Natural-25 variant_id event tier: {variant_id!r}")
    return parts[-2]


def _safe_artifact(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ReleaseValidationError(f"release artifact path must be relative: {relative_path!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ReleaseValidationError(f"release artifact escapes release root: {relative_path!r}") from exc
    if not path.is_file():
        raise ReleaseValidationError(f"missing release artifact: {relative_path}")
    return path


def load_tv2v_source_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load source rows while preserving JSONL order."""
    rows = list(load_jsonl(path))
    if not all(isinstance(row, dict) for row in rows):
        raise ReleaseValidationError(f"{path}: source manifest rows must be objects")
    return rows


def validate_tv2v_source_rows(
    rows: list[dict[str, Any]],
    *,
    local_prompt_rows: list[dict[str, Any]],
    api_prompt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate the 100/100 task-to-static-source bijection."""
    local_none = {
        str(row["variant_id"]): row
        for row in local_prompt_rows
        if row.get("oov_gap") == "none"
    }
    api_static = {
        str(row["variant_id"]): row
        for row in api_prompt_rows
        if row.get("oov_gap") == "static"
    }
    if len(local_none) != 100 or len(api_static) != 100:
        raise ReleaseValidationError(
            f"prompt catalogs must expose 100 local none and 100 API static rows, got {len(local_none)}/{len(api_static)}"
        )
    if len(rows) != 100:
        raise ReleaseValidationError(f"tv2v_sources.jsonl must have 100 rows, got {len(rows)}")

    task_ids: set[str] = set()
    source_catalog_ids: set[str] = set()
    asset_ids: set[str] = set()
    source_video_ids: set[str] = set()
    hf_paths: set[str] = set()
    local_prompts_by_id = {
        str(row["variant_id"]): str(row["ti2v_prompt"])
        for row in local_prompt_rows
    }
    tiers: Counter[str] = Counter()
    for index, row in enumerate(rows):
        row_id = f"source row {index}"
        task_id = str(row.get("task_variant_id") or "")
        source_catalog_id = str(row.get("source_catalog_variant_id") or "")
        expected_source_catalog_id = task_id[: -len("__none")] + "__static" if task_id.endswith("__none") else ""
        if task_id not in local_none:
            raise ReleaseValidationError(f"{row_id}: unknown local task_variant_id {task_id!r}")
        if source_catalog_id != expected_source_catalog_id or source_catalog_id not in api_static:
            raise ReleaseValidationError(
                f"{task_id}: source_catalog_variant_id must be corresponding API static row, got {source_catalog_id!r}"
            )
        if task_id in task_ids or source_catalog_id in source_catalog_ids:
            raise ReleaseValidationError(f"duplicate task/source catalog mapping at {task_id!r}")
        task_ids.add(task_id)
        source_catalog_ids.add(source_catalog_id)

        family_id = str(row.get("family_id") or "")
        if family_id != str(local_none[task_id].get("family_id")) or family_id != str(api_static[source_catalog_id].get("family_id")):
            raise ReleaseValidationError(f"{task_id}: family_id does not agree across catalogs")
        exact_tier = _event_tier(task_id)
        if row.get("event_tier") != exact_tier:
            raise ReleaseValidationError(f"{task_id}: event_tier must be exact {exact_tier!r}")
        tiers[exact_tier] += 1

        if row.get("source_model") != "wan2_7_i2v" or row.get("source_camera") != "static":
            raise ReleaseValidationError(f"{task_id}: source must be wan2_7_i2v/static")
        if row.get("source_video_origin") != "wan2.7_static_camera_tv2v":
            raise ReleaseValidationError(f"{task_id}: invalid source_video_origin")
        if row.get("source_prompt_catalog_id") != "api_source_historical_202606":
            raise ReleaseValidationError(f"{task_id}: unexpected historical source prompt catalog id")
        if row.get("hf_repo") != "WRBench/wrbench-videos":
            raise ReleaseValidationError(f"{task_id}: invalid hf_repo_id")
        if row.get("hf_asset_revision") != VIDEO_ASSET_REVISION:
            raise ReleaseValidationError(f"{task_id}: source asset revision is not pinned to the audited commit")
        source_video_id = str(row.get("source_video_id") or "")
        expected_source_video_id = f"wan2_7_i2v__{task_id}__static"
        if source_video_id != expected_source_video_id or source_video_id in source_video_ids:
            raise ReleaseValidationError(
                f"{task_id}: source_video_id must be unique and equal {expected_source_video_id!r}"
            )
        source_video_ids.add(source_video_id)
        hf_path = str(row.get("hf_path") or "")
        expected_hf_path = f"videos/wan2_7_i2v/static/{source_video_id}.mp4"
        if hf_path != expected_hf_path or hf_path in hf_paths:
            raise ReleaseValidationError(
                f"{task_id}: hf_path must be unique and equal {expected_hf_path!r}"
            )
        hf_paths.add(hf_path)
        asset_id = str(row.get("video_asset_id") or "")
        if not asset_id or asset_id in asset_ids:
            raise ReleaseValidationError(f"{task_id}: video_asset_id must be unique and non-empty")
        asset_ids.add(asset_id)
        asset_sha256 = str(row.get("sha256") or "")
        if not _HEX64_RE.fullmatch(asset_sha256):
            raise ReleaseValidationError(f"{task_id}: invalid asset SHA256")
        if asset_id != f"sha256:{asset_sha256}":
            raise ReleaseValidationError(f"{task_id}: video_asset_id must identify the asset SHA256")
        for field in ("bytes", "decoded_frame_count", "width", "height"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ReleaseValidationError(f"{task_id}: {field} must be a positive integer")
        fps = row.get("fps")
        if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
            raise ReleaseValidationError(f"{task_id}: fps must be positive")
        if row.get("asset_role") != "conditioning_input_not_benchmark_output":
            raise ReleaseValidationError(f"{task_id}: source asset must be labeled as conditioning input")
        policy = str(row.get("temporal_sampling_policy") or "")
        if policy != "owned_by_each_consuming_model_contract":
            raise ReleaseValidationError(f"{task_id}: temporal sampling ownership must remain model-specific")
        exact_prompt = row.get("exact_provider_request_prompt")
        if not isinstance(exact_prompt, str) or not exact_prompt:
            raise ReleaseValidationError(f"{task_id}: exact provider request prompt is missing")
        if row.get("provider_prompt_provenance") != "exact_request_sidecar":
            raise ReleaseValidationError(f"{task_id}: provider prompt provenance must be exact_request_sidecar")
        if row.get("provider_prompt_sha256") != _sha256_text(exact_prompt):
            raise ReleaseValidationError(f"{task_id}: provider_prompt_sha256 does not hash the exact request text")
        if not _HEX64_RE.fullmatch(str(row.get("provider_request_evidence_sha256") or "")):
            raise ReleaseValidationError(f"{task_id}: invalid provider_request_evidence_sha256")

        expected_local_fixed_id = task_id[: -len("__none")] + "__fixed"
        expected_local_fixed_prompt = local_prompts_by_id.get(expected_local_fixed_id)
        if expected_local_fixed_prompt is None:
            raise ReleaseValidationError(
                f"{task_id}: corresponding local fixed prompt {expected_local_fixed_id!r} is missing"
            )
        local_prompt_match = exact_prompt == expected_local_fixed_prompt
        api_prompt_match = exact_prompt == str(api_static[source_catalog_id].get("ti2v_prompt") or "")
        represented = local_prompt_match or api_prompt_match
        expected_represented = exact_tier != "T2_div_a"
        if represented is not expected_represented or local_prompt_match is not expected_represented:
            raise ReleaseValidationError(
                f"{task_id}: exact request does not match the corresponding local fixed prompt/tier policy"
            )
        if exact_tier == "T2_div_a" and api_prompt_match:
            raise ReleaseValidationError(
                f"{task_id}: T2_div_a exact request must not be represented by the API source catalog"
            )
        if row.get("provider_prompt_matches_exact_request_catalog") is not local_prompt_match:
            raise ReleaseValidationError(
                f"{task_id}: provider_prompt_matches_exact_request_catalog does not match the local catalog"
            )
        if row.get("provider_prompt_matches_source_catalog") is not api_prompt_match:
            raise ReleaseValidationError(
                f"{task_id}: provider_prompt_matches_source_catalog does not match the API source catalog"
            )
        expected_status = (
            "exact_catalog_match"
            if represented
            else "exact_request_sidecar_not_represented_by_frozen_catalogs"
        )
        if row.get("provider_prompt_catalog_status") != expected_status:
            raise ReleaseValidationError(
                f"{task_id}: provider prompt catalog status does not match the two frozen catalogs"
            )

        exact_catalog_id = row.get("exact_provider_request_prompt_catalog_id")
        exact_catalog_sha256 = row.get("exact_provider_request_prompt_catalog_sha256")
        exact_variant_id = row.get("exact_provider_request_prompt_variant_id")
        if represented:
            if exact_catalog_id != "local_ti2v_tv2v_paper_generation":
                raise ReleaseValidationError(f"{task_id}: exact request must identify the local fixed catalog")
            if exact_catalog_sha256 != LOCAL_PROMPT_SHA256:
                raise ReleaseValidationError(f"{task_id}: exact provider-request catalog SHA256 mismatch")
            if exact_variant_id != expected_local_fixed_id:
                raise ReleaseValidationError(
                    f"{task_id}: exact provider-request variant must be the corresponding local fixed row"
                )
        elif any(value is not None for value in (exact_catalog_id, exact_catalog_sha256, exact_variant_id)):
            raise ReleaseValidationError(
                f"{task_id}: unrepresented exact request must not claim frozen-catalog metadata"
            )

        source_prompt = str(api_static[source_catalog_id].get("ti2v_prompt") or "")
        if row.get("source_prompt_catalog_sha256") != API_SOURCE_PROMPT_SHA256:
            raise ReleaseValidationError(f"{task_id}: source prompt catalog SHA256 mismatch")
        if row.get("source_prompt_sha256") != _sha256_text(source_prompt):
            raise ReleaseValidationError(f"{task_id}: source_prompt_sha256 does not hash the API static prompt")
        verification = row.get("verification")
        if not isinstance(verification, dict):
            raise ReleaseValidationError(f"{task_id}: verification must be an object")
        expected_verification = {
            "provider_prompt_matches_api_source_catalog": api_prompt_match,
            "provider_prompt_matches_local_fixed_catalog": local_prompt_match,
            "provider_request_model_is_wan2_7_i2v": True,
            "provider_request_prompt_present": True,
            "public_master_bytes_equal": True,
            "public_master_sha256_equal": True,
            "raw_public_bytes_equal": True,
            "raw_public_decoded_metadata_equal": True,
            "raw_public_sha256_equal": True,
        }
        if verification != expected_verification:
            raise ReleaseValidationError(f"{task_id}: verification flags do not match audited provenance")

    if task_ids != set(local_none) or source_catalog_ids != set(api_static):
        raise ReleaseValidationError("source manifest is not a complete 100/100 task-to-static-source bijection")
    if set(tiers) != EVENT_TIERS or any(count != 25 for count in tiers.values()):
        raise ReleaseValidationError(f"source manifest must preserve 25 rows in each exact event tier, got {dict(tiers)}")
    provider_statuses = Counter(str(row["provider_prompt_catalog_status"]) for row in rows)
    if provider_statuses != {
        "exact_catalog_match": 75,
        "exact_request_sidecar_not_represented_by_frozen_catalogs": 25,
    }:
        raise ReleaseValidationError(
            f"unexpected provider request/catalog boundary counts: {dict(provider_statuses)}"
        )
    return {
        "rows": len(rows),
        "event_tiers": dict(sorted(tiers.items())),
        "provider_prompt_catalog_status": dict(sorted(provider_statuses.items())),
    }


def _validate_artifact_ledger(root: Path, manifest: dict[str, Any]) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ReleaseValidationError("release_manifest.json artifacts must be an object keyed by relative path")
    expected_paths = set(NATURAL25_RELEASE_CORE_FILES)
    if set(artifacts) != expected_paths:
        raise ReleaseValidationError(
            f"release artifact ledger mismatch: missing={sorted(expected_paths - set(artifacts))}, "
            f"extra={sorted(set(artifacts) - expected_paths)}"
        )
    for relative_path, record in artifacts.items():
        if not isinstance(record, dict):
            raise ReleaseValidationError(f"artifact ledger entry {relative_path!r} must be an object")
        path = _safe_artifact(root, relative_path)
        if record.get("bytes") != path.stat().st_size:
            raise ReleaseValidationError(f"{relative_path}: artifact byte count mismatch")
        if record.get("sha256") != _sha256_file(path):
            raise ReleaseValidationError(f"{relative_path}: artifact SHA256 mismatch")
        if record.get("row_count") != EXPECTED_ARTIFACT_ROWS[relative_path]:
            raise ReleaseValidationError(f"{relative_path}: artifact logical row_count mismatch")


def _validate_camera_contract(root: Path) -> None:
    scope = _load_json(root / "camera_scope.json")
    if scope.get("schema_version") != "wrbench.paper_camera_scope.v1":
        raise ReleaseValidationError("camera_scope.json schema_version mismatch")
    if scope.get("release_id") != NATURAL25_PAPER_RELEASE_ID:
        raise ReleaseValidationError("camera_scope.json release_id mismatch")
    if scope.get("scope_files") != {
        "api_prompt_camera": "camera_scopes/api_prompt_camera.json",
        "local_dual_angle": "camera_scopes/local_dual_angle.json",
        "local_static": "camera_scopes/local_static.json",
    }:
        raise ReleaseValidationError("camera_scope.json scope_files mismatch")
    if scope.get("wan2_7_source_assets_are_conditioning_inputs") is not True:
        raise ReleaseValidationError("camera_scope.json must label Wan2.7 sources as conditioning inputs")
    rows = scope.get("models")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ReleaseValidationError("camera_scope.json models must be an object list")
    by_model = {str(row.get("model_id")): row for row in rows}
    if len(by_model) != 23 or set(by_model) != set(EXPECTED_MODEL_ROWS):
        raise ReleaseValidationError("camera_scope.json must enumerate exactly the frozen 23 models")
    for model, expected_rows in EXPECTED_MODEL_ROWS.items():
        row = by_model[model]
        if row.get("expected_rows") != expected_rows:
            raise ReleaseValidationError(f"{model}: expected_rows must be {expected_rows}")
        if row.get("prompt_catalog_id") != EXPECTED_MODEL_CATALOG[model]:
            raise ReleaseValidationError(f"{model}: prompt catalog mapping mismatch")
        if model in API_300_MODELS and row.get("requested_yaw_degrees") is not None:
            raise ReleaseValidationError(f"{model}: API prompt-camera rows must not fabricate requested yaw degrees")
        expected_cells = (
            {"static", "yaw_LR_30", "yaw_RL_30", "yaw_LR_60", "yaw_RL_60"}
            if model in LOCAL_500_MODELS
            else {"yaw_LR_30", "yaw_RL_30", "yaw_LR_60", "yaw_RL_60"}
            if model in LOCAL_400_MODELS
            else {"static", "yaw_LR", "yaw_RL"}
        )
        if set(row.get("camera_cells") or []) != expected_cells:
            raise ReleaseValidationError(f"{model}: camera_cells do not match the frozen release")
        if model in (*LOCAL_500_MODELS, *LOCAL_400_MODELS):
            if row.get("requested_yaw_degrees") != [30, 60]:
                raise ReleaseValidationError(f"{model}: local requested yaw degrees must be [30, 60]")
            if row.get("frozen_static_included") is not (model in LOCAL_500_MODELS):
                raise ReleaseValidationError(f"{model}: frozen static inclusion does not match its row group")
            if row.get("current_contract_supports_static") is not EXPECTED_LOCAL_CURRENT_STATIC[model]:
                raise ReleaseValidationError(f"{model}: current static capability does not match the release contract")
    totals = scope.get("totals")
    if totals != {
        "api_prompt_camera": 2100,
        "local_dual_angle": 6400,
        "local_static": 1100,
        "paper_video_rows": 9600,
        "static": 1800,
        "yaw_LR": 3900,
        "yaw_RL": 3900,
    }:
        raise ReleaseValidationError(f"camera_scope.json totals mismatch: {totals!r}")

    local_dual = _load_json(root / "camera_scopes" / "local_dual_angle.json")
    if (
        local_dual.get("schema_version") != "wrbench.camera_scope.v1"
        or local_dual.get("scope_id") != f"{NATURAL25_PAPER_RELEASE_ID}.local_dual_angle"
        or local_dual.get("prompt_catalog_id") != "local_ti2v_tv2v"
        or local_dual.get("rows_per_cell_per_model") != 100
    ):
        raise ReleaseValidationError("local_dual_angle scope metadata mismatch")
    if set(local_dual.get("models", [])) != set((*LOCAL_500_MODELS, *LOCAL_400_MODELS)):
        raise ReleaseValidationError("local_dual_angle scope must enumerate the exact 16 local models")
    local_dual_cells = {
        (cell.get("camera_type"), cell.get("requested_yaw_deg"))
        for cell in local_dual.get("cells", [])
        if isinstance(cell, dict)
    }
    if local_dual_cells != EXPECTED_LOCAL_DUAL_CELLS or len(local_dual.get("cells", [])) != 4:
        raise ReleaseValidationError("local_dual_angle scope must contain the exact LR/RL x 30/60 cells")
    if local_dual.get("expected_rows") != 6400:
        raise ReleaseValidationError("local_dual_angle expected_rows must be 6400")
    local_static = _load_json(root / "camera_scopes" / "local_static.json")
    if (
        local_static.get("schema_version") != "wrbench.camera_scope.v1"
        or local_static.get("scope_id") != f"{NATURAL25_PAPER_RELEASE_ID}.local_static"
        or local_static.get("prompt_catalog_id") != "local_ti2v_tv2v"
        or local_static.get("rows_per_model") != 100
        or local_static.get("camera_type") != "static"
    ):
        raise ReleaseValidationError("local_static scope metadata mismatch")
    if set(local_static.get("models", [])) != set(LOCAL_500_MODELS) or local_static.get("expected_rows") != 1100:
        raise ReleaseValidationError("local_static scope must enumerate the exact 11 models / 1,100 rows")
    exclusions = local_static.get("excluded_models")
    if not isinstance(exclusions, list) or not all(isinstance(row, dict) for row in exclusions):
        raise ReleaseValidationError("local_static excluded_models must be an object list")
    exclusions_by_model = {str(row.get("model_id")): row for row in exclusions}
    if set(exclusions_by_model) != set(LOCAL_400_MODELS) or len(exclusions_by_model) != 5:
        raise ReleaseValidationError("local_static scope must enumerate the exact five frozen exclusions")
    for model, exclusion in exclusions_by_model.items():
        if exclusion.get("current_contract_supports_static") is not EXPECTED_LOCAL_CURRENT_STATIC[model]:
            raise ReleaseValidationError(f"{model}: excluded-model current static capability mismatch")
        if exclusion.get("frozen_omission") != by_model[model].get("frozen_static_omission"):
            raise ReleaseValidationError(f"{model}: frozen static omission wording is inconsistent")
    gamecraft_omission = (
        "historical frozen-release fact; the freeze-time reason is not established by retained evidence"
    )
    if exclusions_by_model["hunyuan_game_craft"].get("frozen_omission") != gamecraft_omission:
        raise ReleaseValidationError("Hunyuan GameCraft freeze-time static omission must remain unresolved")
    api_scope = _load_json(root / "camera_scopes" / "api_prompt_camera.json")
    if (
        api_scope.get("schema_version") != "wrbench.camera_scope.v1"
        or api_scope.get("scope_id") != f"{NATURAL25_PAPER_RELEASE_ID}.api_prompt_camera"
        or api_scope.get("prompt_catalog_id") != "api_source"
        or api_scope.get("rows_per_cell_per_model") != 100
        or api_scope.get("control_semantics")
        != "prompt-camera intent; no requested 30/60-degree distinction and no fabricated target C2W"
    ):
        raise ReleaseValidationError("api_prompt_camera scope metadata mismatch")
    if set(api_scope.get("models", [])) != set(API_300_MODELS):
        raise ReleaseValidationError("api_prompt_camera scope must enumerate the exact seven API models")
    api_cells = {
        (cell.get("camera_type"), cell.get("requested_yaw_deg"))
        for cell in api_scope.get("cells", [])
        if isinstance(cell, dict)
    }
    if api_cells != EXPECTED_API_CELLS or len(api_scope.get("cells", [])) != 3:
        raise ReleaseValidationError("api_prompt_camera scope must contain exact static/LR/RL intent cells")
    if api_scope.get("expected_rows") != 2100:
        raise ReleaseValidationError("api_prompt_camera expected_rows must be 2100")
    for cell in api_scope.get("cells", []):
        if cell.get("requested_yaw_deg") is not None:
            raise ReleaseValidationError("API prompt-camera cells must have null requested_yaw_deg")


def _validate_prompt_usage(root: Path) -> None:
    usage = _load_json(root / "prompt_usage.json")
    if usage.get("schema_version") != "wrbench.prompt_usage.v1":
        raise ReleaseValidationError("prompt_usage.json schema_version mismatch")
    if usage.get("release_id") != NATURAL25_PAPER_RELEASE_ID:
        raise ReleaseValidationError("prompt_usage.json release_id mismatch")
    if usage.get("catalogs") != {
        "api_source": {
            "path": "variants.api_source.jsonl",
            "rows": 400,
            "sha256": API_SOURCE_PROMPT_SHA256,
        },
        "local_ti2v_tv2v": {
            "path": "variants.local_ti2v_tv2v.jsonl",
            "rows": 400,
            "sha256": LOCAL_PROMPT_SHA256,
        },
    }:
        raise ReleaseValidationError("prompt_usage.json catalog ledger mismatch")
    if usage.get("local_generation") != {
        "camera_control": "supplied separately from content prompts",
        "catalog_id": "local_ti2v_tv2v",
        "historical_catalog_rows_not_all_sent": 300,
        "selected_oov_gap": "none",
        "selected_rows": 100,
    }:
        raise ReleaseValidationError("prompt_usage.json local-generation semantics mismatch")
    if usage.get("api_prompt_camera") != {
        "camera_to_source_oov_gap": {
            "static": "static",
            "yaw_LR": "yaw_LR",
            "yaw_RL": "yaw_RL",
        },
        "catalog_id": "api_source",
        "provider_request_boundary": (
            "The catalog stores historical source prompts. A provider-specific prompt_to_send "
            "is exact only when request evidence exists."
        ),
        "selected_rows": 300,
    }:
        raise ReleaseValidationError("prompt_usage.json API prompt-camera semantics mismatch")
    if usage.get("tv2v_conditioning_source_requests") != {
        "exact_provider_request_sidecars": 100,
        "exact_requests_matching_a_frozen_catalog": 75,
        "exact_requests_not_represented_by_frozen_catalogs": 25,
        "mapping_boundary": (
            "task_variant_id to source_catalog_variant_id identifies the conditioning asset; "
            "it does not assert provider request prompt-text equality"
        ),
        "not_represented_event_tier": "T2_div_a",
    }:
        raise ReleaseValidationError("prompt_usage.json TV2V request-boundary semantics mismatch")
    if usage.get("model_to_catalog") != EXPECTED_MODEL_CATALOG:
        raise ReleaseValidationError("prompt_usage.json must map all 23 models to exactly one catalog")
    if usage.get("expected_output_rows_by_model") != EXPECTED_MODEL_ROWS:
        raise ReleaseValidationError("prompt_usage.json expected output rows do not reproduce 500/400/300 groups")


def _validate_hydra_policy(root: Path) -> None:
    policy = _load_json(root / "hydra_evaluation_policy.json")
    if policy.get("segments") != {
        "generated_continuation": {"start_inclusive": 77, "stop_exclusive": 154},
        "source_condition": {"start_inclusive": 0, "stop_exclusive": 77},
    }:
        raise ReleaseValidationError("HyDRA segments must be [0,77) source and [77,154) generated")
    metrics = policy.get("metric_surfaces")
    if not isinstance(metrics, dict):
        raise ReleaseValidationError("HyDRA metric_surfaces must be an object")
    d1 = metrics.get("D1_current_evaluator_policy")
    if d1 != {"crop_before_pose_inference": True, "frames": [77, 154]}:
        raise ReleaseValidationError("HyDRA D1 current evaluator policy must crop [77,154) before pose inference")
    if metrics.get("D2_D6_current_evaluator_policy") != {"frames": [0, 154]}:
        raise ReleaseValidationError("HyDRA D2-D6 policy must score the full 154-frame submission")
    if policy.get("published_D1_provenance", {}).get("CamPrec") != "legacy_full_concat_pose_then_posthoc_slice":
        raise ReleaseValidationError("HyDRA legacy CamPrec provenance must remain explicit")


def _reject_mutable_only_links(value: Any, *, location: str = "manifest") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_mutable_only_links(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_mutable_only_links(child, location=f"{location}[{index}]")
    elif isinstance(value, str) and ("/resolve/main/" in value or "/blob/main/" in value):
        raise ReleaseValidationError(f"{location}: mutable main link is not version-pinned")


def validate_natural25_release(
    release_id: str = NATURAL25_PAPER_RELEASE_ID,
    *,
    release_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate hashes, joins, camera scope, and HyDRA semantics for a release."""
    root = Path(release_dir).expanduser().resolve() if release_dir is not None else natural25_release_dir(release_id).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"release directory not found: {root}")
    manifest = _load_json(root / "release_manifest.json")
    if manifest.get("schema_version") != "wrbench.paper_release.v1" or manifest.get("release_id") != release_id:
        raise ReleaseValidationError("release manifest schema_version/release_id mismatch")
    if manifest.get("paper_model_count") != 23 or manifest.get("paper_video_rows") != 9600:
        raise ReleaseValidationError("release manifest must freeze 23 models / 9,600 paper rows")
    if manifest.get("current_video_dataset_rows") != 11100:
        raise ReleaseValidationError("release manifest must distinguish the expanded 11,100-row public dataset")
    if manifest.get("scores_changed") is not False or manifest.get("video_bytes_changed") is not False:
        raise ReleaseValidationError("provenance repair must declare scores_changed=false and video_bytes_changed=false")
    if manifest.get("source_video_assets", {}).get("revision") != VIDEO_ASSET_REVISION:
        raise ReleaseValidationError("release manifest source-video revision mismatch")
    if manifest.get("model_to_prompt_catalog") != EXPECTED_MODEL_CATALOG:
        raise ReleaseValidationError("release manifest must map all frozen models to exactly one prompt catalog")
    for field in ("baseline_commits", "producer_commits"):
        revisions = manifest.get(field)
        if not isinstance(revisions, dict) or not revisions:
            raise ReleaseValidationError(f"release manifest must record {field}")
        for revision in revisions.values():
            if not _HEX40_RE.fullmatch(str(revision)):
                raise ReleaseValidationError(f"{field} entry is not immutable: {revision!r}")
    _reject_mutable_only_links(manifest)
    _validate_artifact_ledger(root, manifest)

    local_path = root / "variants.local_ti2v_tv2v.jsonl"
    api_path = root / "variants.api_source.jsonl"
    if _sha256_file(local_path) != LOCAL_PROMPT_SHA256 or _sha256_file(api_path) != API_SOURCE_PROMPT_SHA256:
        raise ReleaseValidationError("paper prompt catalog checksum mismatch")
    local_rows = list(load_jsonl(local_path))
    api_rows = list(load_jsonl(api_path))
    if len(local_rows) != 400 or len(api_rows) != 400:
        raise ReleaseValidationError("paper prompt catalogs must each contain 400 rows")
    if release_dir is None and _sha256_file(natural25_variants_path()) != TOOLKIT_VARIANTS_SHA256:
        raise ReleaseValidationError("compatibility variants.jsonl changed unexpectedly")

    _validate_prompt_usage(root)

    source_summary = validate_tv2v_source_rows(
        load_tv2v_source_rows(root / "tv2v_sources.jsonl"),
        local_prompt_rows=local_rows,
        api_prompt_rows=api_rows,
    )
    _validate_camera_contract(root)
    _validate_hydra_policy(root)
    return {
        "release_id": release_id,
        "paper_model_count": 23,
        "paper_video_rows": 9600,
        "prompt_rows": {"api_source": 400, "local_ti2v_tv2v": 400},
        "source_rows": source_summary,
        "status": "ok",
    }
