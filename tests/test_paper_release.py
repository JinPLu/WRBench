from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest

from wrbench.datasets import (
    NATURAL25_PAPER_RELEASE_ID,
    NATURAL25_RELEASE_CORE_FILES,
    available_natural25_releases,
    load_jsonl,
    load_natural25_release_index,
    load_natural25_release_manifest,
    natural25_release_dir,
    natural25_release_path,
)
from wrbench.release_validation import (
    API_SOURCE_PROMPT_SHA256,
    EXPECTED_MODEL_CATALOG,
    EXPECTED_MODEL_ROWS,
    LOCAL_PROMPT_SHA256,
    ReleaseValidationError,
    VIDEO_ASSET_REVISION,
    validate_natural25_release,
    validate_tv2v_source_rows,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(relative_path: str) -> dict:
    return json.loads(natural25_release_path(relative_path).read_text(encoding="utf-8"))


def test_paper_release_validates_as_one_self_consistent_contract() -> None:
    summary = validate_natural25_release()

    assert summary["status"] == "ok"
    assert summary["release_id"] == NATURAL25_PAPER_RELEASE_ID
    assert summary["paper_model_count"] == 23
    assert summary["paper_video_rows"] == 9600
    assert summary["source_rows"]["rows"] == 100


def test_exact_prompt_catalog_rows_and_fixed_sha256() -> None:
    local_path = natural25_release_path("variants.local_ti2v_tv2v.jsonl")
    api_path = natural25_release_path("variants.api_source.jsonl")

    assert len(list(load_jsonl(local_path))) == 400
    assert len(list(load_jsonl(api_path))) == 400
    assert _sha256(local_path) == LOCAL_PROMPT_SHA256
    assert _sha256(api_path) == API_SOURCE_PROMPT_SHA256


def test_source_manifest_is_exact_task_to_static_bijection_with_request_boundary() -> None:
    local_rows = list(load_jsonl(natural25_release_path("variants.local_ti2v_tv2v.jsonl")))
    api_rows = list(load_jsonl(natural25_release_path("variants.api_source.jsonl")))
    source_rows = list(load_jsonl(natural25_release_path("tv2v_sources.jsonl")))
    local_task_ids = {row["variant_id"] for row in local_rows if row["oov_gap"] == "none"}
    api_static_ids = {row["variant_id"] for row in api_rows if row["oov_gap"] == "static"}

    assert len(source_rows) == 100
    assert {row["task_variant_id"] for row in source_rows} == local_task_ids
    assert {row["source_catalog_variant_id"] for row in source_rows} == api_static_ids
    assert all(
        row["source_catalog_variant_id"]
        == row["task_variant_id"][: -len("__none")] + "__static"
        for row in source_rows
    )
    assert Counter(row["event_tier"] for row in source_rows) == {
        "T0": 25,
        "T1": 25,
        "T2_div_a": 25,
        "T2_div_b": 25,
    }
    assert Counter(row["provider_prompt_catalog_status"] for row in source_rows) == {
        "exact_catalog_match": 75,
        "exact_request_sidecar_not_represented_by_frozen_catalogs": 25,
    }
    assert all(row["hf_asset_revision"] == VIDEO_ASSET_REVISION for row in source_rows)
    assert all(row["verification"]["raw_public_sha256_equal"] is True for row in source_rows)
    assert all(row["asset_role"] == "conditioning_input_not_benchmark_output" for row in source_rows)


def test_source_manifest_validator_hashes_exact_request_text() -> None:
    local_rows = list(load_jsonl(natural25_release_path("variants.local_ti2v_tv2v.jsonl")))
    api_rows = list(load_jsonl(natural25_release_path("variants.api_source.jsonl")))
    source_rows = [dict(row) for row in load_jsonl(natural25_release_path("tv2v_sources.jsonl"))]
    source_rows[0]["provider_prompt_sha256"] = "0" * 64

    with pytest.raises(ReleaseValidationError, match="does not hash the exact request text"):
        validate_tv2v_source_rows(
            source_rows,
            local_prompt_rows=local_rows,
            api_prompt_rows=api_rows,
        )


def test_source_manifest_validator_recomputes_catalog_representation() -> None:
    local_rows = list(load_jsonl(natural25_release_path("variants.local_ti2v_tv2v.jsonl")))
    api_rows = list(load_jsonl(natural25_release_path("variants.api_source.jsonl")))
    source_rows = [dict(row) for row in load_jsonl(natural25_release_path("tv2v_sources.jsonl"))]
    represented = next(
        row for row in source_rows if row["provider_prompt_catalog_status"] == "exact_catalog_match"
    )
    represented["exact_provider_request_prompt"] += " Deliberately altered outside both catalogs."
    represented["provider_prompt_sha256"] = hashlib.sha256(
        represented["exact_provider_request_prompt"].encode("utf-8")
    ).hexdigest()

    with pytest.raises(
        ReleaseValidationError,
        match="does not match the corresponding local fixed prompt/tier policy",
    ):
        validate_tv2v_source_rows(
            source_rows,
            local_prompt_rows=local_rows,
            api_prompt_rows=api_rows,
        )


def test_source_manifest_validator_binds_exact_request_to_its_own_task() -> None:
    local_rows = list(load_jsonl(natural25_release_path("variants.local_ti2v_tv2v.jsonl")))
    api_rows = list(load_jsonl(natural25_release_path("variants.api_source.jsonl")))
    source_rows = [dict(row) for row in load_jsonl(natural25_release_path("tv2v_sources.jsonl"))]
    represented = next(row for row in source_rows if row["event_tier"] == "T0")
    unrepresented = next(row for row in source_rows if row["event_tier"] == "T2_div_a")
    request_fields = (
        "exact_provider_request_prompt",
        "exact_provider_request_prompt_catalog_id",
        "exact_provider_request_prompt_catalog_sha256",
        "exact_provider_request_prompt_variant_id",
        "provider_prompt_catalog_status",
        "provider_prompt_matches_exact_request_catalog",
        "provider_prompt_matches_source_catalog",
        "provider_prompt_provenance",
        "provider_prompt_sha256",
        "provider_request_evidence_sha256",
        "verification",
    )
    for field in request_fields:
        represented[field], unrepresented[field] = unrepresented[field], represented[field]

    with pytest.raises(
        ReleaseValidationError,
        match="does not match the corresponding local fixed prompt/tier policy",
    ):
        validate_tv2v_source_rows(
            source_rows,
            local_prompt_rows=local_rows,
            api_prompt_rows=api_rows,
        )


def test_camera_scope_reproduces_frozen_9600_and_api_has_no_degrees() -> None:
    camera = _json("camera_scope.json")
    api = _json("camera_scopes/api_prompt_camera.json")

    assert camera["totals"] == {
        "api_prompt_camera": 2100,
        "local_dual_angle": 6400,
        "local_static": 1100,
        "paper_video_rows": 9600,
        "static": 1800,
        "yaw_LR": 3900,
        "yaw_RL": 3900,
    }
    assert all(cell["requested_yaw_deg"] is None for cell in api["cells"])
    api_models = {row["model_id"]: row for row in camera["models"] if row["input_surface"] == "api_prompt_camera"}
    assert len(api_models) == 7
    assert all(row["requested_yaw_degrees"] is None for row in api_models.values())


def test_release_validator_rejects_camera_scope_model_substitution(tmp_path: Path) -> None:
    release_copy = tmp_path / NATURAL25_PAPER_RELEASE_ID
    shutil.copytree(natural25_release_dir(), release_copy)
    relative_path = "camera_scopes/local_static.json"
    scope_path = release_copy / relative_path
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    scope["models"][0] = "gen3c"
    scope_path.write_text(json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = release_copy / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][relative_path]["bytes"] = scope_path.stat().st_size
    manifest["artifacts"][relative_path]["sha256"] = _sha256(scope_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="exact 11 models"):
        validate_natural25_release(release_dir=release_copy)


def test_all_23_models_map_to_one_catalog_and_500_400_300_rows() -> None:
    usage = _json("prompt_usage.json")

    assert usage["model_to_catalog"] == EXPECTED_MODEL_CATALOG
    assert usage["expected_output_rows_by_model"] == EXPECTED_MODEL_ROWS
    assert Counter(usage["expected_output_rows_by_model"].values()) == {500: 11, 400: 5, 300: 7}
    assert sum(usage["expected_output_rows_by_model"].values()) == 9600


def test_hydra_policy_separates_current_evaluator_and_legacy_camprec() -> None:
    policy = _json("hydra_evaluation_policy.json")

    assert policy["segments"]["source_condition"] == {"start_inclusive": 0, "stop_exclusive": 77}
    assert policy["segments"]["generated_continuation"] == {"start_inclusive": 77, "stop_exclusive": 154}
    assert policy["metric_surfaces"]["D1_current_evaluator_policy"] == {
        "crop_before_pose_inference": True,
        "frames": [77, 154],
    }
    assert policy["metric_surfaces"]["D2_D6_current_evaluator_policy"] == {"frames": [0, 154]}
    assert policy["published_D1_provenance"]["CamPrec"] == "legacy_full_concat_pose_then_posthoc_slice"


def test_release_files_and_manifest_load_from_installed_package_layout() -> None:
    assert NATURAL25_PAPER_RELEASE_ID in available_natural25_releases()
    release_root = natural25_release_dir()
    assert release_root.is_dir()
    assert all(natural25_release_path(path).is_file() for path in NATURAL25_RELEASE_CORE_FILES)
    manifest = load_natural25_release_manifest()
    assert manifest["scores_changed"] is False
    assert manifest["video_bytes_changed"] is False
    assert manifest["current_video_dataset_rows"] == 11100


def test_release_index_records_live_cross_repository_publication() -> None:
    index = load_natural25_release_index()

    assert index["github"] == {
        "contract_commit": "6c47fc6dbcdc7b90ea83e8aaf0f038035d933614",
        "release_tag": "v0.1.1",
        "repository": "JinPLu/WRBench",
    }
    assert index["hugging_face"]["immutable_tag"] == "paper-main-20260608-repro-v1"
    assert index["hugging_face"]["natural25"]["commit"] == "214ed8cd5cb3494bcfe332c06fa1db8bdff9edd8"
    assert index["hugging_face"]["videos"]["commit"] == "2de5487a6ac0e5d1f551a8d6e1c83b9e00f73d66"
    assert index["hugging_face"]["natural25"]["configs"] == {
        "paper_tv2v_sources": {"columns": 38, "rows": 100},
        "paper_variants_api_source": {"columns": 9, "rows": 400},
        "paper_variants_local": {"columns": 9, "rows": 400},
    }
    assert index["hugging_face"]["videos"]["configs"] == {
        "paper_camera_scope": {"columns": 25, "rows": 23},
        "tv2v_source_videos": {"columns": 46, "rows": 100},
        "videos_master": {"columns": 45, "rows": 11100},
    }
    for surface in ("natural25", "videos"):
        assert index["hugging_face"][surface]["viewer"] == {
            "failed_jobs": 0,
            "first_rows_readable": True,
            "pending_jobs": 0,
            "splits_readable": True,
        }
    expected_hashes = index["artifact_hashes"]
    assert expected_hashes["release_manifest.json"] == _sha256(
        natural25_release_path("release_manifest.json")
    )
    for relative_path in (
        "README.md",
        "camera_scope.json",
        "camera_scopes/api_prompt_camera.json",
        "camera_scopes/local_dual_angle.json",
        "camera_scopes/local_static.json",
        "hydra_evaluation_policy.json",
        "prompt_usage.json",
        "tv2v_sources.jsonl",
        "variants.api_source.jsonl",
        "variants.local_ti2v_tv2v.jsonl",
    ):
        assert expected_hashes[relative_path] == _sha256(natural25_release_path(relative_path))
    assert index["verification"] == {
        "paper_model_count": 23,
        "paper_video_rows": 9600,
        "scores_changed": False,
        "source_assets_revision": VIDEO_ASSET_REVISION,
        "verified_at_utc": "2026-07-11T19:58:31Z",
        "video_bytes_changed": False,
    }
