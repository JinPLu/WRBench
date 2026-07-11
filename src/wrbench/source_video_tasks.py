"""Strict Natural-25 source-video task maps.

Paper-release source assets are keyed by the content task variant (the
``...__none`` row).  Camera-bearing output ids and coarser family/reasoning
keys are deliberately not accepted because one conditioning asset is reused
across all camera cells for that task.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EVENT_TIERS = {"T0", "T1", "T2_div_a", "T2_div_b"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPEATED_FIRST_FRAME_ORIGINS = {
    "first_frame_loop",
    "first_frame_repeat",
    "repeated_first_frame",
}


class SourceVideoTaskMapError(ValueError):
    """Raised when a source-video task map is incomplete or unsafe."""


def _require_text(row: dict[str, Any], field: str, *, row_id: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SourceVideoTaskMapError(f"{row_id}: missing non-empty {field}")
    return value.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _event_tier_from_task_variant_id(task_variant_id: str) -> str:
    parts = task_variant_id.split("__")
    if len(parts) < 3 or parts[-1] != "none" or parts[-2] not in EVENT_TIERS:
        raise SourceVideoTaskMapError(
            f"task_variant_id must end with an exact event tier and '__none': {task_variant_id!r}"
        )
    return parts[-2]


@dataclass(frozen=True)
class SourceVideoBinding:
    """One verified source-video binding for a Natural-25 content task."""

    task_variant_id: str
    source_catalog_variant_id: str
    source_video_id: str
    source_video_path: Path
    source_video_sha256: str
    source_video_bytes: int
    source_video_origin: str
    source_video_usage: str
    event_tier: str
    hf_repo_id: str
    hf_path: str
    hf_revision: str
    task_map_path: Path

    def compile_kwargs(self) -> dict[str, str]:
        return {"source_video": str(self.source_video_path)}

    def manifest_metadata(self) -> dict[str, Any]:
        return {
            "source_video_task_variant_id": self.task_variant_id,
            "source_catalog_variant_id": self.source_catalog_variant_id,
            "source_video_id": self.source_video_id,
            "source_video_path": str(self.source_video_path),
            "source_video_sha256": self.source_video_sha256,
            "source_video_bytes": self.source_video_bytes,
            "source_video_origin": self.source_video_origin,
            "source_video_usage": self.source_video_usage,
            "source_video_event_tier": self.event_tier,
            "source_video_hf_repo_id": self.hf_repo_id,
            "source_video_hf_path": self.hf_path,
            "source_video_asset_revision": self.hf_revision,
            "source_video_task_map_path": str(self.task_map_path),
        }

    def sidecar_provenance(self) -> dict[str, Any]:
        metadata = self.manifest_metadata()
        metadata.pop("source_video_task_map_path")
        return metadata


class SourceVideoTaskMap:
    """Resolve and checksum explicit source videos by ``task_variant_id`` only."""

    def __init__(self, *, path: Path, source_root: Path, rows: list[dict[str, Any]]) -> None:
        self.path = path.resolve()
        self.source_root = source_root.expanduser().resolve()
        if not self.source_root.is_dir():
            raise FileNotFoundError(f"source-video root not found: {self.source_root}")
        self._by_task: dict[str, dict[str, Any]] = {}
        self._verified: set[tuple[Path, str, int]] = set()
        for index, row in enumerate(rows):
            row_id = f"{self.path}: row {index}"
            task_variant_id = _require_text(row, "task_variant_id", row_id=row_id)
            if task_variant_id in self._by_task:
                raise SourceVideoTaskMapError(f"{self.path}: duplicate task_variant_id {task_variant_id!r}")
            event_tier = _event_tier_from_task_variant_id(task_variant_id)
            if _require_text(row, "event_tier", row_id=task_variant_id) != event_tier:
                raise SourceVideoTaskMapError(
                    f"{task_variant_id}: event_tier must be exact {event_tier!r}"
                )
            expected_source_id = task_variant_id[: -len("__none")] + "__static"
            if _require_text(row, "source_catalog_variant_id", row_id=task_variant_id) != expected_source_id:
                raise SourceVideoTaskMapError(
                    f"{task_variant_id}: source_catalog_variant_id must be {expected_source_id!r}"
                )
            self._by_task[task_variant_id] = dict(row)
        if not self._by_task:
            raise SourceVideoTaskMapError(f"{self.path}: source-video task map is empty")

    def _source_path(self, row: dict[str, Any], *, task_variant_id: str) -> Path:
        raw = row.get("source_video")
        if not isinstance(raw, str) or not raw.strip():
            raise SourceVideoTaskMapError(f"{task_variant_id}: missing source_video relative path")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise SourceVideoTaskMapError(
                f"{task_variant_id}: source_video must be relative to --source-video-root"
            )
        path = (self.source_root / relative).resolve()
        try:
            path.relative_to(self.source_root)
        except ValueError as exc:
            raise SourceVideoTaskMapError(
                f"{task_variant_id}: source_video escapes --source-video-root"
            ) from exc
        return path

    def _verify(self, path: Path, *, sha256: str, byte_count: int, task_variant_id: str) -> None:
        cache_key = (path, sha256, byte_count)
        if cache_key in self._verified:
            return
        if not path.is_file():
            raise FileNotFoundError(f"{task_variant_id}: source video not found: {path}")
        actual_bytes = path.stat().st_size
        if actual_bytes != byte_count:
            raise SourceVideoTaskMapError(
                f"{task_variant_id}: source video byte count {actual_bytes} != expected {byte_count}: {path}"
            )
        actual_sha256 = _sha256_file(path)
        if actual_sha256 != sha256:
            raise SourceVideoTaskMapError(
                f"{task_variant_id}: source video SHA256 {actual_sha256} != expected {sha256}: {path}"
            )
        self._verified.add(cache_key)

    def resolve(
        self,
        *,
        model: str,
        task_variant_id: str,
        source_video_usage: str,
    ) -> SourceVideoBinding:
        row = self._by_task.get(task_variant_id)
        if row is None:
            raise SourceVideoTaskMapError(
                f"{model}: source-video task map {self.path} has no task_variant_id={task_variant_id!r}"
            )
        origin = _require_text(row, "source_video_origin", row_id=task_variant_id)
        normalized_origin = origin.lower().replace("-", "_")
        if bool(row.get("source_video_is_repeated_first_frame")) or normalized_origin in _REPEATED_FIRST_FRAME_ORIGINS:
            raise SourceVideoTaskMapError(
                f"{model}: {task_variant_id} uses a repeated first-frame clip; "
                "this is not a valid WRBench benchmark source-video input"
            )
        sha256 = str(row.get("source_video_sha256") or row.get("sha256") or "").lower()
        if not _SHA256_RE.fullmatch(sha256):
            raise SourceVideoTaskMapError(f"{task_variant_id}: missing valid source video SHA256")
        raw_bytes = row.get("source_video_bytes", row.get("bytes"))
        if not isinstance(raw_bytes, int) or isinstance(raw_bytes, bool) or raw_bytes <= 0:
            raise SourceVideoTaskMapError(f"{task_variant_id}: source video bytes must be a positive integer")
        source_path = self._source_path(row, task_variant_id=task_variant_id)
        self._verify(source_path, sha256=sha256, byte_count=raw_bytes, task_variant_id=task_variant_id)
        return SourceVideoBinding(
            task_variant_id=task_variant_id,
            source_catalog_variant_id=_require_text(row, "source_catalog_variant_id", row_id=task_variant_id),
            source_video_id=_require_text(row, "source_video_id", row_id=task_variant_id),
            source_video_path=source_path,
            source_video_sha256=sha256,
            source_video_bytes=raw_bytes,
            source_video_origin=origin,
            source_video_usage=source_video_usage,
            event_tier=_require_text(row, "event_tier", row_id=task_variant_id),
            hf_repo_id=_require_text(row, "hf_repo_id", row_id=task_variant_id),
            hf_path=_require_text(row, "hf_path", row_id=task_variant_id),
            hf_revision=_require_text(row, "hf_revision", row_id=task_variant_id),
            task_map_path=self.path,
        )


def _rows_from_payload(payload: Any, *, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
        rows = payload["tasks"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise SourceVideoTaskMapError(f"{path}: source-video task map must be a list or an object with tasks[]")
    if not all(isinstance(row, dict) for row in rows):
        raise SourceVideoTaskMapError(f"{path}: every source-video task row must be an object")
    return [dict(row) for row in rows]


def load_source_video_task_map(
    path: str | Path,
    *,
    source_root: str | Path,
) -> SourceVideoTaskMap:
    """Load a strict task map whose source paths are relative to ``source_root``."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"source-video task map not found: {resolved}")
    if resolved.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in resolved.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = _rows_from_payload(json.loads(resolved.read_text(encoding="utf-8")), path=resolved)
    return SourceVideoTaskMap(path=resolved, source_root=Path(source_root), rows=rows)
