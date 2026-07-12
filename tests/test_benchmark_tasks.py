from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from wrbench.benchmark import (
    NATURAL25_CAMERA_COMBOS,
    load_natural25_camera_scope,
    natural25_camera_tasks,
    natural25_camera_tasks_from_scope,
)
from wrbench.datasets import (
    PROMPT_PROFILE_T2V_LAYOUT_ANCHOR,
    PROMPT_PROFILE_TI2V_ACTIVE,
    load_jsonl,
    load_natural25_t2v_event_tails,
    load_natural25_t2v_layout_anchors,
    natural25_t2v_event_tails_path,
    natural25_prompt_profile_path,
    natural25_release_path,
    natural25_t2v_rotation_stress_camera_scope_path,
    published_t2v_results_json,
)


def test_natural25_camera_benchmark_expands_to_500_outputs() -> None:
    tasks = natural25_camera_tasks()

    assert len(tasks) == 500
    assert len({task.output_id for task in tasks}) == 500
    assert len({task.variant_id for task in tasks}) == 100
    assert len({task.family_id for task in tasks}) == 25
    assert Counter(task.camera for task in tasks) == {camera: 100 for camera in NATURAL25_CAMERA_COMBOS}


def test_natural25_camera_benchmark_can_filter_cameras() -> None:
    tasks = natural25_camera_tasks(cameras=["yaw_LR", "yaw_RL"])

    assert len(tasks) == 200
    assert set(task.camera for task in tasks) == {"yaw_LR", "yaw_RL"}


def test_t2v_rotation_stress_scope_is_the_only_shipped_rotation_scope_snapshot() -> None:
    scope_path = natural25_t2v_rotation_stress_camera_scope_path()

    assert scope_path.is_file()
    assert not (scope_path.parent / "rotation_stress_30_60.json").exists()


def test_t2v_event_tails_are_separate_from_ti2v_prompt_records() -> None:
    tails = load_natural25_t2v_event_tails()
    tail_rows = list(load_jsonl(natural25_t2v_event_tails_path()))

    assert len(tails) == 400
    assert len(tail_rows) == 400
    assert tails["bedroom_adult_bed_sit__T1__none"] == "Immediately, the adult person walks toward the bed."
    assert not any(". is out of frame" in tail for tail in tails.values())


def test_natural25_rotation_stress_scope_expands_to_500_outputs_without_pan() -> None:
    scope = load_natural25_camera_scope(natural25_t2v_rotation_stress_camera_scope_path())
    tasks = natural25_camera_tasks_from_scope(camera_scope=scope)

    assert scope.scope_id == "natural25_t2v_rotation_stress_30_60"
    assert len(tasks) == 500
    assert len({task.output_id for task in tasks}) == 500
    assert len({task.variant_id for task in tasks}) == 100
    assert Counter(task.camera for task in tasks) == {
        "static": 100,
        "yaw30_LR": 100,
        "yaw30_RL": 100,
        "yaw60_LR": 100,
        "yaw60_RL": 100,
    }
    assert Counter(task.camera_type for task in tasks) == {"static": 100, "yaw_LR": 200, "yaw_RL": 200}
    assert Counter(task.stress_yaw_deg for task in tasks) == {None: 100, 30.0: 200, 60.0: 200}
    assert all("pan" not in task.camera for task in tasks)


def test_t2v_results_metadata_uses_separate_prompt_and_scope() -> None:
    payload = json.loads(published_t2v_results_json().read_text(encoding="utf-8"))
    definition = payload["benchmark_definition"]

    assert payload["table_policy"] == "separate_from_frozen_23model_main_table"
    assert payload["prompt_of_record"] == "src/wrbench/data/natural25/prompt_profiles/t2v_layout_anchor.json"
    assert payload["frozen_main_table_prompt_contract"].endswith(
        "releases/paper_main_20260608/prompt_usage.json"
    )
    assert "legacy_main_table_prompt" not in payload
    assert definition["leaderboard_track"] == "t2v_rotation_stress"
    assert definition["prompt_profile_id"] == "t2v_layout_anchor"
    assert definition["camera_scope_id"] == "natural25_t2v_rotation_stress_30_60"
    assert definition["camera_scope_path"].endswith("camera_scopes/t2v_rotation_stress_30_60.json")
    assert definition["camera_controls"] == ["static", "yaw30_LR", "yaw30_RL", "yaw60_LR", "yaw60_RL"]
    assert all("pan" not in camera for camera in definition["camera_controls"])
    assert payload["models"][0]["generation"]["camera_scope_id"] == definition["camera_scope_id"]
    assert payload["models"][0]["generation"]["prompt_profile_id"] == definition["prompt_profile_id"]


def test_paper_local_dual_angle_scope_is_runner_compatible() -> None:
    scope = load_natural25_camera_scope(
        natural25_release_path("camera_scopes/local_dual_angle.json")
    )
    tasks = natural25_camera_tasks_from_scope(
        camera_scope=scope,
        variants_path=natural25_release_path("variants.local_ti2v_tv2v.jsonl"),
    )

    assert len(scope.applicable_models or ()) == 16
    assert len(tasks) == 400
    assert Counter(task.camera for task in tasks) == {
        "yaw_LR_30": 100,
        "yaw_RL_30": 100,
        "yaw_LR_60": 100,
        "yaw_RL_60": 100,
    }
    assert Counter(task.stress_yaw_deg for task in tasks) == {30.0: 200, 60.0: 200}


def test_paper_local_static_scope_is_runner_compatible() -> None:
    scope = load_natural25_camera_scope(
        natural25_release_path("camera_scopes/local_static.json")
    )
    tasks = natural25_camera_tasks_from_scope(
        camera_scope=scope,
        variants_path=natural25_release_path("variants.local_ti2v_tv2v.jsonl"),
    )

    assert len(scope.applicable_models or ()) == 11
    assert len(tasks) == 100
    assert {task.camera for task in tasks} == {"static"}
    assert {task.stress_yaw_deg for task in tasks} == {None}


def test_versioned_scope_loader_does_not_depend_on_release_id(tmp_path: Path) -> None:
    source = natural25_release_path("camera_scopes/local_static.json")
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["scope_id"] = "example_release.local_static"
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    scope = load_natural25_camera_scope(path)

    assert scope.scope_id == "example_release.local_static"
    assert scope.expected_task_count == 100


def test_paper_api_prompt_camera_scope_is_explicitly_inspect_only() -> None:
    with pytest.raises(ValueError, match="inspect-only"):
        load_natural25_camera_scope(
            natural25_release_path("camera_scopes/api_prompt_camera.json")
        )


def test_t2v_layout_anchor_prompt_profile_keeps_ti2v_prompt_as_record() -> None:
    scope = load_natural25_camera_scope(natural25_t2v_rotation_stress_camera_scope_path())
    default_tasks = natural25_camera_tasks_from_scope(camera_scope=scope)
    t2v_tasks = natural25_camera_tasks_from_scope(
        camera_scope=scope,
        prompt_profile=PROMPT_PROFILE_T2V_LAYOUT_ANCHOR,
    )

    default_sample = next(
        task
        for task in default_tasks
        if task.variant_id == "bedroom_adult_bed_sit__T1__none" and task.camera == "static"
    )
    t2v_sample = next(
        task
        for task in t2v_tasks
        if task.variant_id == "bedroom_adult_bed_sit__T1__none" and task.camera == "static"
    )

    assert default_sample.prompt_profile_id == PROMPT_PROFILE_TI2V_ACTIVE
    assert default_sample.prompt == default_sample.ti2v_prompt
    assert t2v_sample.prompt_profile_id == PROMPT_PROFILE_T2V_LAYOUT_ANCHOR
    assert t2v_sample.prompt != t2v_sample.ti2v_prompt
    assert "Realistic photography" not in t2v_sample.prompt
    assert "left third of the frame" in t2v_sample.prompt
    assert "far on the right side of the frame" in t2v_sample.prompt
    assert "Immediately, the adult person walks toward the bed" in t2v_sample.prompt


def test_t2v_layout_anchor_prompt_profile_covers_rotation_stress_variants() -> None:
    scope = load_natural25_camera_scope(natural25_t2v_rotation_stress_camera_scope_path())
    tasks = natural25_camera_tasks_from_scope(
        camera_scope=scope,
        prompt_profile=PROMPT_PROFILE_T2V_LAYOUT_ANCHOR,
    )
    anchors = load_natural25_t2v_layout_anchors()
    profile = json.loads(natural25_prompt_profile_path(PROMPT_PROFILE_T2V_LAYOUT_ANCHOR).read_text())
    prompts_by_variant: dict[str, set[str]] = {}

    assert len(tasks) == 500
    assert len({task.variant_id for task in tasks}) == 100
    for task in tasks:
        prompts_by_variant.setdefault(task.variant_id, set()).add(task.prompt)
        anchor = anchors[task.family_id]
        assert anchor["subject"].rstrip(".") in task.prompt
        assert anchor["interactor"].rstrip(".") in task.prompt
        assert anchor["open_surface"].rstrip(".") in task.prompt
        assert "left third of the frame" in task.prompt
        assert "far on the right side of the frame" in task.prompt
        assert "camera slowly moves" not in task.prompt
        prompt_lower = task.prompt.lower()
        for token in profile["banned_style_tokens"]:
            assert token.lower() not in prompt_lower

    assert all(len(prompts) == 1 for prompts in prompts_by_variant.values())
