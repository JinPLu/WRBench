from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wrbench.paper_sources import SourcePreparationError, prepare_paper_tv2v_sources
from wrbench.source_video_tasks import load_source_video_task_map


TASK_ID = "bedroom_adult_bed_sit__T0__none"


def _row(payload: bytes) -> dict:
    return {
        "bytes": len(payload),
        "decoded_frame_count": 81,
        "event_tier": "T0",
        "family_id": "bedroom_adult_bed_sit",
        "fps": 16.0,
        "height": 720,
        "hf_path": f"videos/wan2_7_i2v/static/wan2_7_i2v__{TASK_ID}__static.mp4",
        "hf_repo_id": "WRBench/wrbench-videos",
        "hf_revision": "8a927b9322c5d8af6474399ce4840ef4148f8e39",
        "local_relpath": "wan2_7_i2v/static/source.mp4",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "source_catalog_variant_id": "bedroom_adult_bed_sit__T0__static",
        "source_video_id": f"wan2_7_i2v__{TASK_ID}__static",
        "source_video_origin": "wan2.7_static_camera_tv2v",
        "task_variant_id": TASK_ID,
        "video_asset_id": "asset-source",
        "width": 1280,
    }


def test_preparation_plan_is_side_effect_free(tmp_path: Path) -> None:
    source_root = tmp_path / "not-created"
    summary = prepare_paper_tv2v_sources(
        [_row(b"asset")],
        release_id="paper_main_20260608",
        source_root=source_root,
        mode="plan",
    )

    assert summary["results"][0]["status"] == "would_download"
    assert not source_root.exists()


def test_preparation_resumes_partial_download_and_task_map_is_runner_compatible(tmp_path: Path) -> None:
    payload = b"complete-canonical-video"
    row = _row(payload)
    source_root = tmp_path / "sources"
    part = source_root / "wan2_7_i2v" / "static" / "source.mp4.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(payload[:8])
    calls: list[tuple[str, bool]] = []

    def downloader(url: str, part_path: Path, resume: bool) -> None:
        calls.append((url, resume))
        assert part_path == part
        with part_path.open("ab") as fh:
            fh.write(payload[8:])

    summary = prepare_paper_tv2v_sources(
        [row],
        release_id="paper_main_20260608",
        source_root=source_root,
        mode="download",
        resume=True,
        downloader=downloader,
    )

    assert summary["results"][0]["status"] == "downloaded"
    assert calls == [(summary["results"][0]["url"], True)]
    task_map_path = source_root / "source_video_task_map.json"
    task_map_payload = json.loads(task_map_path.read_text(encoding="utf-8"))
    assert task_map_payload["tasks"][0]["task_variant_id"] == TASK_ID
    task_map = load_source_video_task_map(task_map_path, source_root=source_root)
    binding = task_map.resolve(
        model="hydra",
        task_variant_id=TASK_ID,
        source_video_usage="temporal_conditioning_source_video",
    )
    assert binding.source_video_path.read_bytes() == payload


def test_preparation_reuses_verified_download_and_rejects_corruption(tmp_path: Path) -> None:
    payload = b"complete-canonical-video"
    row = _row(payload)
    source_root = tmp_path / "sources"
    destination = source_root / row["local_relpath"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(payload)

    def should_not_download(_url: str, _part_path: Path, _resume: bool) -> None:
        raise AssertionError("verified existing file must not be downloaded")

    summary = prepare_paper_tv2v_sources(
        [row],
        release_id="paper_main_20260608",
        source_root=source_root,
        mode="download",
        resume=True,
        downloader=should_not_download,
    )
    assert summary["results"][0]["status"] == "verified_existing"

    destination.write_bytes(b"corrupt")
    with pytest.raises(SourcePreparationError, match="refusing to overwrite corrupt"):
        prepare_paper_tv2v_sources(
            [row],
            release_id="paper_main_20260608",
            source_root=source_root,
            mode="download",
            resume=True,
            downloader=should_not_download,
        )
