from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import wrbench.source_video_tasks as source_tasks
from wrbench.source_video_tasks import SourceVideoTaskMapError, load_source_video_task_map


TASK_ID = "bedroom_adult_bed_sit__T0__none"


def _row(payload: bytes, *, origin: str = "wan2.7_static_camera_tv2v") -> dict:
    return {
        "bytes": len(payload),
        "event_tier": "T0",
        "hf_path": f"videos/wan2_7_i2v/static/wan2_7_i2v__{TASK_ID}__static.mp4",
        "hf_repo_id": "WRBench/wrbench-videos",
        "hf_revision": "8a927b9322c5d8af6474399ce4840ef4148f8e39",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_catalog_variant_id": "bedroom_adult_bed_sit__T0__static",
        "source_video": "wan2_7_i2v/static/source.mp4",
        "source_video_id": f"wan2_7_i2v__{TASK_ID}__static",
        "source_video_origin": origin,
        "task_variant_id": TASK_ID,
    }


def _load(tmp_path: Path, rows: list[dict], payload: bytes = b"canonical-source"):
    source_root = tmp_path / "sources"
    source_path = source_root / "wan2_7_i2v" / "static" / "source.mp4"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(payload)
    map_path = tmp_path / "task_map.json"
    map_path.write_text(json.dumps({"tasks": rows}), encoding="utf-8")
    return load_source_video_task_map(map_path, source_root=source_root)


def test_source_video_task_map_resolves_only_exact_task_variant_id(tmp_path: Path) -> None:
    payload = b"canonical-source"
    task_map = _load(tmp_path, [_row(payload)], payload)

    binding = task_map.resolve(
        model="hydra",
        task_variant_id=TASK_ID,
        source_video_usage="temporal_conditioning_source_video",
    )

    assert binding.task_variant_id == TASK_ID
    assert binding.source_catalog_variant_id == "bedroom_adult_bed_sit__T0__static"
    assert binding.source_video_path.read_bytes() == payload
    with pytest.raises(SourceVideoTaskMapError, match="has no task_variant_id"):
        task_map.resolve(
            model="hydra",
            task_variant_id="bedroom_adult_bed_sit__T1__none",
            source_video_usage="temporal_conditioning_source_video",
        )


def test_source_video_task_map_rejects_duplicate_task_variants(tmp_path: Path) -> None:
    payload = b"canonical-source"
    with pytest.raises(SourceVideoTaskMapError, match="duplicate task_variant_id"):
        _load(tmp_path, [_row(payload), _row(payload)], payload)


def test_source_video_task_map_rejects_repeated_first_frame_clip(tmp_path: Path) -> None:
    payload = b"canonical-source"
    row = _row(payload, origin="repeated_first_frame")
    row["source_video_is_repeated_first_frame"] = True
    task_map = _load(tmp_path, [row], payload)

    with pytest.raises(SourceVideoTaskMapError, match="repeated first-frame clip"):
        task_map.resolve(
            model="hydra",
            task_variant_id=TASK_ID,
            source_video_usage="temporal_conditioning_source_video",
        )


def test_source_video_checksum_verification_is_cached_across_camera_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"canonical-source"
    task_map = _load(tmp_path, [_row(payload)], payload)
    calls = 0
    original = source_tasks._sha256_file

    def counted(path: Path) -> str:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(source_tasks, "_sha256_file", counted)
    for _camera_cell in ("yaw_LR_30", "yaw_LR_60", "yaw_RL_30", "yaw_RL_60"):
        task_map.resolve(
            model="hydra",
            task_variant_id=TASK_ID,
            source_video_usage="temporal_conditioning_source_video",
        )

    assert calls == 1
