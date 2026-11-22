"""Tests for D2 full-FoV visual-integrity extraction and contract alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from wrbench.eval.aggregate import latest_d1_d6_metrics as metric_contract
from wrbench.eval.aggregate.build_main_table import build_table
from wrbench.eval.d2 import extract_d2_dinov2_fullfov_candidate as d2_fullfov


class _SizeLike:
    def __init__(self, **values: object) -> None:
        self._values = values

    def items(self):
        return self._values.items()


class _SizeDictLike:
    def __init__(self, **values: object) -> None:
        self._values = values
        for key, value in values.items():
            setattr(self, key, value)

    def get(self, key: str) -> object:
        return self._values.get(key)


def test_resize_preserves_aspect_and_padding_reaches_patch_multiple() -> None:
    resized = d2_fullfov.resize_preserve_aspect_shape(100, 200, 224)
    padded = d2_fullfov.pad_shape_to_patch_multiple(*resized, patch_size=14)

    assert resized == (224, 448)
    assert padded == (224, 448)
    assert padded[0] % 14 == 0
    assert padded[1] % 14 == 0
    assert d2_fullfov.resize_preserve_aspect_shape(704, 1280, 256) == (256, 465)
    assert d2_fullfov.pad_shape_to_patch_multiple(256, 465, 14) == (266, 476)


def test_valid_patch_mask_excludes_batch_common_padding_only() -> None:
    mask = d2_fullfov.valid_patch_mask_for_shape((15, 15), (28, 42), patch_size=14)

    assert mask.tolist() == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    assert int(np.count_nonzero(mask)) == 4


def test_preprocess_normalizes_before_zero_padding_in_tensor_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResampled:
        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        def __array__(self, dtype=None) -> np.ndarray:
            return np.asarray(self._arr, dtype=dtype)

    class _FakeImage:
        Resampling = SimpleNamespace(BICUBIC=3)
        BICUBIC = 3

        def __init__(self, arr: np.ndarray) -> None:
            self._arr = arr

        @classmethod
        def fromarray(cls, arr: np.ndarray) -> "_FakeImage":
            return cls(arr)

        def resize(self, size: tuple[int, int], resample: int = 3) -> _FakeResampled:
            assert resample == 3
            w, h = size
            out = np.zeros((h, w, 3), dtype=np.uint8)
            return _FakeResampled(out)

    monkeypatch.setitem(sys.modules, "PIL", SimpleNamespace(Image=_FakeImage))
    norm_config = {
        "image_mean": [0.485, 0.456, 0.406],
        "image_std": [0.229, 0.224, 0.225],
        "do_rescale": True,
        "rescale_factor": 1.0 / 255.0,
        "resize_short_edge": 10,
    }
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    batch, masks, metas, batch_meta = d2_fullfov.preprocess_fullfov_batch(
        [frame],
        norm_config=norm_config,
        patch_size=14,
    )

    assert batch.shape == (1, 3, 14, 14)
    # Content pixels [0..10, 0..10] hold normalized black (-mean/std), while padded pixels are 0.0.
    assert batch[0, 0, 0, 0] == pytest.approx(-0.485 / 0.229, abs=1e-5)
    assert np.all(batch[0, :, 10:, :] == 0.0)
    assert np.all(batch[0, :, :, 10:] == 0.0)
    assert masks.tolist() == [[True]]
    assert metas[0]["d2_lg_v2_valid_patch_count"] == 1
    assert batch_meta["d2_lg_v2_batch_common_padded_shape"] == [14, 14]


def test_processor_norm_config_reads_sizedict_shortest_edge() -> None:
    class Processor:
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.2, 0.2, 0.2]
        do_rescale = True
        rescale_factor = 1.0 / 255.0
        size = _SizeDictLike(
            height=None,
            width=None,
            longest_edge=None,
            shortest_edge=np.int64(256),
            max_height=None,
            max_width=None,
        )
        crop_size = _SizeDictLike(height=np.int64(224), width=np.int64(224))
        do_center_crop = True

    got = d2_fullfov._processor_norm_config(Processor())

    assert got["resize_short_edge"] == 256
    assert got["processor_size"] == {"shortest_edge": 256}
    assert got["processor_crop_size"] == {"height": 224, "width": 224}


def test_local_bestmatch_and_candidate_e_ignore_padding_tokens() -> None:
    global_features = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )
    patch_features = np.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, -1.0]],
            [[1.0, 0.0], [0.5, 0.5]],
        ],
        dtype=np.float32,
    )
    patch_masks = np.asarray(
        [
            [True, False],
            [True, False],
            [True, False],
        ]
    )

    got = d2_fullfov.compute_fullfov_candidate(global_features, patch_features, patch_masks)

    assert got["d2_lg_v2_local_low_p20"] == 1.0
    assert got["d2_lg_v2_g_first_last"] == pytest.approx(0.6, abs=1e-6)
    assert got["d2_lg_v2_candidate_e"] == pytest.approx(0.6, abs=1e-6)
    assert all(not key.startswith("d2_lg_") or key.startswith("d2_lg_v2_") for key in got)


def test_d2_checks_manifest_asset_hash_before_scoring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import hashlib

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"frozen video bytes")
    row = {"video_id": "published", "video_path": str(clip), "sha256": "0" * 64}
    with pytest.raises(ValueError, match="Video SHA-256 mismatch for published"):
        d2_fullfov._extract_one(row, None, None, "cpu", **d2_fullfov.PAPER_SAMPLING)

    monkeypatch.setattr(d2_fullfov, "_sample_frames", lambda *a, **kw: (None, {}))
    row["sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
    result = d2_fullfov._extract_one(row, None, None, "cpu", **d2_fullfov.PAPER_SAMPLING)
    assert result["d2_lg_v2_video_sha256"] == row["sha256"]
    assert result["d2_lg_v2_paper_sampling"] is True


def test_merge_candidate_fields_populates_selected_score_and_table_integration() -> None:
    global_features = np.asarray([[1.0, 0.0], [0.6, 0.8]], dtype=np.float32)
    patch_features = np.asarray(
        [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, -1.0]]],
        dtype=np.float32,
    )
    patch_masks = np.asarray([[True, False], [True, False]])
    candidate = d2_fullfov.compute_fullfov_candidate(global_features, patch_features, patch_masks)

    row = d2_fullfov.merge_candidate_fields(
        {"video_id": "v1", "model": "gen3c", "feature_status": "ok"},
        candidate,
    )
    failed = d2_fullfov.merge_candidate_fields(
        {"video_id": "v2", "model": "gen3c", "feature_status": "decode_failed"},
        d2_fullfov.compute_fullfov_candidate(np.empty((0, 1)), np.empty((0, 1, 1)), np.empty((0, 1), dtype=bool)),
    )

    assert row["d2_status"] == "ok"
    assert row["d2_selected_visual_integrity_score"] == pytest.approx(0.6, abs=1e-6)
    assert "d2_dinov2_temporal_consistency" not in row
    assert failed["d2_status"] == "decode_failed"
    assert failed["d2_selected_visual_integrity_score"] is None

    table, _ = build_table(runtime_records=[], d1_records=[], d2_records=[row, failed])
    assert table[0]["D2_visual_integrity"] == pytest.approx(0.6, abs=1e-6)
    assert table[0]["D2_visual_integrity_n"] == 1
    assert table[0]["D2_visual_integrity_eligible"] == 2

    qa = d2_fullfov.qa_rows([row, failed], requested=2)
    assert qa["all_ok_selected_scores_match_candidate_e"] is True
    assert qa["forbidden_mainline_present"] is False
    assert qa["missing_audit_fields"] == []


def test_frame_indices_time_fps_retains_endpoints_and_caps_max_frames() -> None:
    idx = d2_fullfov._frame_indices_for_sampling(
        source_frame_count=81,
        source_fps=16.0,
        sample_policy="time_fps",
        sample_fps=3.0,
        min_frames=4,
        max_frames=24,
    )
    assert idx[0] == 0
    assert idx[-1] == 80
    assert len(idx) == 16
    assert idx == sorted(set(idx))


def test_interpolate_pos_encoding_detection_handles_supported_and_unsupported() -> None:
    class Supported:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, pixel_values, interpolate_pos_encoding: bool = False):
            return {"interp": interpolate_pos_encoding, "shape": tuple(pixel_values.shape)}

    class Unsupported:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

        def forward(self, pixel_values):
            return {"interp": None, "shape": tuple(pixel_values.shape)}

    arr = np.zeros((1, 3, 14, 14), dtype=np.float32)
    assert d2_fullfov._supports_interpolate_pos_encoding(Supported()) is True
    assert d2_fullfov._supports_interpolate_pos_encoding(Unsupported()) is False
    assert d2_fullfov._model_forward(Supported(), arr, interpolate_pos_encoding=True)["interp"] is True
    assert d2_fullfov._model_forward(Unsupported(), arr, interpolate_pos_encoding=True)["interp"] is None


def test_runtime_example_and_contract_match_fullfov_3fps_max24(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from wrbench.eval import runtime as eval_runtime_mod

    repo_root = Path(__file__).resolve().parents[1]
    example_cfg = eval_runtime_mod.load_eval_runtime(repo_root / "wrbench.runtime.example.json")
    assert example_cfg is not None
    sampling = eval_runtime_mod._d2_sampling(example_cfg.scorers)

    assert sampling["sample_policy"] == "time_fps"
    assert sampling["sample_fps"] == 3.0
    assert sampling["min_frames"] == 4
    assert sampling["max_frames"] == 24
    assert sampling["batch_size"] == 4

    d2_spec = metric_contract.metric_by_dimension("D2")
    assert d2_spec.owner == "wrbench.eval.d2.extract_d2_dinov2_fullfov_candidate"

    py = tmp_path / "python"
    py.write_text("#!/bin/sh\n", encoding="utf-8")
    py.chmod(0o755)
    model_dir = tmp_path / "dinov2"
    model_dir.mkdir()
    captured: list[tuple[str, str, list[str]]] = []

    def fake_run(python_bin: str, module: str, argv: list[str], *, env=None) -> int:
        captured.append((python_bin, module, argv))
        return 0

    monkeypatch.setattr(eval_runtime_mod, "run_scorer_python", fake_run)
    cfg = eval_runtime_mod.EvalRuntimeConfig(
        schema_version=1,
        scorers=eval_runtime_mod.EvalScorerRuntime(
            gpu_id=0,
            dinov2_python_bin=str(py),
            dinov2_model_path=str(model_dir),
        ),
    )
    rc = eval_runtime_mod.d2_extract(
        eval_runtime=cfg,
        videos_manifest=tmp_path / "videos.jsonl",
        out_jsonl=tmp_path / "d2.jsonl",
    )
    assert rc == 0
    assert captured[0][1] == "wrbench.eval.d2.extract_d2_dinov2_fullfov_candidate"
    assert "--sample-fps" in captured[0][2] and "3.0" in captured[0][2]
    assert "--max-frames" in captured[0][2] and "24" in captured[0][2]


def test_video_rows_resolves_relative_paths_and_concat_surface(tmp_path: Path) -> None:
    manifest = tmp_path / "videos.json"
    manifest.write_text(
        json.dumps(
            [
                {"video_id": "v_rel", "path": "clips/a.mp4"},
                {
                    "video_id": "v_concat",
                    "evaluated_segment": "generated_only",
                    "eval_video_path": "clips/gen_only.mp4",
                    "original_concat_path": "clips/full_concat.mp4",
                },
            ]
        ),
        encoding="utf-8",
    )
    rows = d2_fullfov._video_rows(manifest, video_root=tmp_path / "root")
    by_id = {row["video_id"]: row for row in rows}
    assert by_id["v_rel"]["video_path"] == str(tmp_path / "root" / "clips" / "a.mp4")
    assert by_id["v_concat"]["video_path"] == str(tmp_path / "root" / "clips" / "full_concat.mp4")
