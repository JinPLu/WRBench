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
    FIRST_FRAME_GENERATION_CATALOG_SHA256,
    ReleaseValidationError,
    VIDEO_ASSET_REVISION,
    validate_natural25_release,
    validate_tv2v_source_rows,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(relative_path: str) -> dict:
    return json.loads(natural25_release_path(relative_path).read_text(encoding="utf-8"))


def _copy_natural25_release_fixture(tmp_path: Path) -> tuple[Path, Path]:
    natural25_copy = tmp_path / "natural25"
    release_copy = natural25_copy / "releases" / NATURAL25_PAPER_RELEASE_ID
    shutil.copytree(natural25_release_dir(), release_copy)
    source_natural25 = natural25_release_dir().parents[1]
    shutil.copy2(source_natural25 / "first_frames_manifest.json", natural25_copy)
    shutil.copytree(source_natural25 / "first_frames", natural25_copy / "first_frames")
    return natural25_copy, release_copy


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


def test_first_frame_catalog_manifest_and_image_bytes_are_exact() -> None:
    catalog_path = natural25_release_path("first_frame_generation_families.jsonl")
    rows = list(load_jsonl(catalog_path))
    manifest_path = natural25_release_dir().parents[1] / "first_frames_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_family = {row["family_id"]: row for row in rows}

    assert len(rows) == len(manifest) == 25
    assert _sha256(catalog_path) == FIRST_FRAME_GENERATION_CATALOG_SHA256
    assert {row["family_id"] for row in manifest} == set(by_family)
    for row in manifest:
        prompt = by_family[row["family_id"]]["t2i_scene"]
        image_path = natural25_release_dir().parents[1] / row["image_path"]
        assert row["t2i_scene"] == prompt
        assert row["t2i_scene_sha256"] == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        assert row["image_sha256"] == _sha256(image_path)
        assert row["image_asset_id"] == f"sha256:{row['image_sha256']}"


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
    _, release_copy = _copy_natural25_release_fixture(tmp_path)
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


def test_release_validator_rejects_unbound_first_frame_manifest(tmp_path: Path) -> None:
    _, release_copy = _copy_natural25_release_fixture(tmp_path)
    manifest_path = release_copy / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["first_frame_surface"]["image_manifest_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="image manifest SHA256 mismatch"):
        validate_natural25_release(release_dir=release_copy)


def test_release_validator_rejects_swapped_first_frame_assets(tmp_path: Path) -> None:
    natural25_copy, release_copy = _copy_natural25_release_fixture(tmp_path)
    image_manifest_path = natural25_copy / "first_frames_manifest.json"
    image_manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    asset_fields = ("image_path", "image_sha256", "image_asset_id")
    first_asset = {field: image_manifest[0][field] for field in asset_fields}
    for field in asset_fields:
        image_manifest[0][field] = image_manifest[1][field]
        image_manifest[1][field] = first_asset[field]
    image_manifest_path.write_text(
        json.dumps(image_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    release_manifest_path = release_copy / "release_manifest.json"
    release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
    release_manifest["first_frame_surface"]["image_manifest_sha256"] = _sha256(image_manifest_path)
    release_manifest_path.write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ReleaseValidationError, match="canonical image_path"):
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
    assert manifest["change_flags_scope"] == "frozen_paper_table_and_historical_attestation_only"
    assert manifest["rolling_release_boundary"] == {
        "does_not_rewrite_frozen_paper_aggregates": True,
        "may_correct_paper_associated_per_video_metadata": True,
        "may_replace_verified_assets_and_rescore": True,
        "separate_versioned_surface": True,
    }
    assert manifest["intended_publication"] == {
        "github_release_tag": "v0.1.2",
        "hf_natural25_provenance_tag": "paper-main-20260608-repro-v2",
        "hf_videos_frozen_attestation_tag": "paper-main-20260608-repro-v1",
        "hf_videos_rolling_update_tag": "rolling-main-20260712-v2",
        "release_index_path_after_hf_publication": "release_index.json",
    }
    assert manifest["current_video_dataset_rows"] == 11100


def test_release_index_records_live_cross_repository_publication() -> None:
    index = load_natural25_release_index()

    assert index["github"] == {
        "contract_commit": "7691d0b51c299f19661d6f90394b05e117cac175",
        "release_tag": "v0.1.2",
        "release_tag_target": "publication_envelope_commit",
        "repository": "JinPLu/WRBench",
    }
    assert index["hugging_face"]["natural25"] == {
        "commit": "65572c4a4b2fe5e71d9195d98ba57a2e4ea78b10",
        "immutable_tag": "paper-main-20260608-repro-v2",
        "release_root": "original/releases/paper_main_20260608",
        "repository": "WRBench/wrbench-natural25",
    }
    assert index["hugging_face"]["videos"] == {
        "current_main_commit": "a8e3d38b3c588a6f5368bfb42bfcf7d9a7dddf89",
        "frozen_attestation": {
            "commit": "b37c39b4b75c8a4420ba6131cb599089192ce443",
            "immutable_tag": "paper-main-20260608-repro-v1",
        },
        "repository": "WRBench/wrbench-videos",
        "rolling_update": {
            "commit": "a8e3d38b3c588a6f5368bfb42bfcf7d9a7dddf89",
            "status": "published_and_anonymously_verified",
            "tag": "rolling-main-20260713-v3",
        },
    }
    expected_hashes = index["artifact_hashes"]
    assert expected_hashes["release_manifest.json"] == _sha256(
        natural25_release_path("release_manifest.json")
    )
    for relative_path in NATURAL25_RELEASE_CORE_FILES:
        assert expected_hashes[relative_path] == _sha256(natural25_release_path(relative_path))
    assert index["external_artifact_hashes"] == {
        "../../first_frames_manifest.json": _sha256(
            natural25_release_dir().parents[1] / "first_frames_manifest.json"
        )
    }
    assert index["verification"] == {
        "paper_model_count": 23,
        "paper_video_rows": 9600,
        "scores_changed": False,
        "scores_changed_scope": "frozen_paper_aggregate_table",
        "rolling_per_video_score_correction": {
            "fields": ["D5_returned_spatial", "D6_returned_state"],
            "rows": 2073,
        },
        "rolling_model_score_sync": {
            "aggregate_results_commit": "30d2245a3d6a7043b7f369cb6666ce3d24aa017e",
            "aggregate_scores_changed": False,
            "field_diff_counts": {
                "D3_visible_spatial": 58,
                "D4_visible_state": 58,
                "D5_returned_spatial": 10,
                "D6_returned_state": 11,
            },
            "model": "easyanimate-v51-camera",
            "replaced_snapshot_sha256": (
                "63dc6c374438d8985b987f1e0a69dc34041bf8eef91276f1e3e62eb66e13fae9"
            ),
            "rows": 59,
            "source_snapshot_sha256": (
                "d4f5632205f91ed913aca25a44549c7772dd9b6b342e85fd83079189afb30bf0"
            ),
        },
        "source_assets_revision": VIDEO_ASSET_REVISION,
        "verified_at_utc": "2026-07-12T19:06:54Z",
        "video_bytes_changed": False,
    }
