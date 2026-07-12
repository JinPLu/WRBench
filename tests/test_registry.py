"""Tests for wrbench.registry: single-source-of-truth invariants."""

import json

from pathlib import Path

import pytest

import wrbench.adapters  # ensure all adapter modules are imported and registered
from wrbench.adapters.base import adapter_for_model, registered_model_keys
from wrbench.registry import (
    MODELS_DIR,
    VALID_INPUT_KINDS,
    VALID_TRANSLATION_UNITS,
    RegistryError,
    active_model_keys,
    all_records,
    canonical_model_key,
    deferred_model_keys,
    model_record,
)


# ---------------------------------------------------------------------------
# Every JSON in models/ loads without RegistryError.
# ---------------------------------------------------------------------------

def test_all_json_load():
    records = all_records()
    assert len(records) > 0, "No model records loaded from models/"


@pytest.mark.parametrize("json_path", sorted(MODELS_DIR.glob("*.json")),
                         ids=lambda p: p.stem)
def test_json_loads_without_error(json_path):
    import json
    from wrbench.registry import _parse_record
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    record = _parse_record(json_path, payload)
    assert record.key


# ---------------------------------------------------------------------------
# canonical_model_key resolves each alias to the canonical key.
# ---------------------------------------------------------------------------

def test_canonical_key_resolves_alias_to_self():
    for record in all_records():
        for alias in record.aliases:
            resolved = canonical_model_key(alias)
            assert resolved == record.key, (
                f"alias {alias!r} resolved to {resolved!r} instead of {record.key!r}"
            )


def test_canonical_key_case_insensitive():
    for record in all_records():
        upper = record.key.upper()
        assert canonical_model_key(upper) == record.key


def test_canonical_key_underscore_dash_interchangeable():
    for record in all_records():
        with_underscores = record.key.replace("-", "_")
        assert canonical_model_key(with_underscores) == record.key


def test_unknown_key_raises():
    with pytest.raises(KeyError, match="Unknown model key or alias"):
        canonical_model_key("this-model-does-not-exist")


# ---------------------------------------------------------------------------
# Each ACTIVE record has valid input_kind, non-empty adapter, valid translation_unit.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", active_model_keys(), ids=active_model_keys())
def test_active_record_input_kind(key):
    record = model_record(key)
    assert record.input_kind in VALID_INPUT_KINDS, (
        f"{key}: input_kind {record.input_kind!r} not in {VALID_INPUT_KINDS}"
    )


def test_source_video_model_input_classification_is_explicit():
    expected = {
        "gen3c": ("TV2V", "temporal_conditioning_via_gen3c_vipe_cache"),
        "hydra": ("TV2V", "temporal_conditioning_source_video"),
        "inspatio-world": ("TV2V", "temporal_conditioning_source_video"),
        "recammaster": ("TV2V", "temporal_conditioning_source_video"),
        "spatia": ("TI2V", "first_frame_extraction"),
        "liveworld": ("TI2V", "first_frame_extraction"),
    }
    for key, (model_input, source_video_usage) in expected.items():
        record = model_record(key)
        assert record.input_kind == "source_video"
        assert record.model_input == model_input
        assert record.source_video_usage == source_video_usage


def test_paper_result_models_resolve_to_registry_metadata():
    results_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wrbench"
        / "data"
        / "results"
        / "wrbench_23model_results.json"
    )
    rows = json.loads(results_path.read_text(encoding="utf-8"))["rows"]

    assert len(rows) == 23
    for row in rows:
        record = model_record(row["model_id"])
        assert record.model_input == row["model_input"]
        assert record.viewpoint_condition_type == row["viewpoint_condition_type"]


def test_frozen_static_exclusions_match_current_registry_capabilities():
    expected = {
        "gen3c": False,
        "hunyuan-game-craft": True,
        "hydra": False,
        "inspatio-world": False,
        "recammaster": False,
    }
    for key, supports_static in expected.items():
        assert model_record(key).capabilities["supports_static"] is supports_static


@pytest.mark.parametrize("key", active_model_keys(), ids=active_model_keys())
def test_active_record_adapter_nonempty(key):
    record = model_record(key)
    assert record.adapter, f"{key}: adapter must be non-empty"


@pytest.mark.parametrize("key", active_model_keys(), ids=active_model_keys())
def test_active_record_translation_unit(key):
    record = model_record(key)
    assert record.amplitude.translation_unit in VALID_TRANSLATION_UNITS, (
        f"{key}: translation_unit {record.amplitude.translation_unit!r} not valid"
    )


# ---------------------------------------------------------------------------
# For every active model, an adapter is registered and has a .compile attribute.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", active_model_keys(), ids=active_model_keys())
def test_active_model_has_registered_adapter(key):
    assert key in registered_model_keys(), (
        f"{key}: no adapter registered; registered keys: {registered_model_keys()}"
    )


@pytest.mark.parametrize("key", active_model_keys(), ids=active_model_keys())
def test_adapter_for_model_has_compile(key):
    adapter = adapter_for_model(key)
    assert hasattr(adapter, "compile"), f"{key}: adapter missing .compile"
    assert callable(adapter.compile), f"{key}: adapter.compile is not callable"


# ---------------------------------------------------------------------------
# Deferred models are reported and adapter_for_model raises for them.
# ---------------------------------------------------------------------------

def test_deferred_model_keys_returns_list():
    deferred = deferred_model_keys()
    assert isinstance(deferred, list)


def test_deferred_models_not_in_active():
    active = set(active_model_keys())
    for key in deferred_model_keys():
        assert key not in active, f"{key} is both deferred and active"


@pytest.mark.parametrize("key", deferred_model_keys() or ["_no_deferred_models_"],
                         ids=deferred_model_keys() or ["_no_deferred_models_"])
def test_adapter_for_deferred_raises(key):
    if key == "_no_deferred_models_":
        pytest.skip("no deferred models present")
    with pytest.raises(ValueError, match="deferred"):
        adapter_for_model(key)
