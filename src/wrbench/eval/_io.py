"""Shared JSON/JSONL, sharding, hashing, and video path utilities for WRBench eval."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def safe_video_id(video_id: Any) -> str:
    """Return a filesystem-safe identifier fragment."""
    return str(video_id or "sample").replace("/", "_")


def to_jsonable(value: Any) -> Any:
    """Recursively coerce numpy/Path values into JSON-serializable Python types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and not isinstance(value, Mapping):
        return to_jsonable(value.tolist())
    if hasattr(value, "item") and not isinstance(value, Mapping):
        return to_jsonable(value.item())
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return str(value)


def read_json(path: str | Path) -> Any:
    """Load arbitrary JSON payload from ``path``."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_records(path: str | Path, *, allow_missing: bool = False, strict: bool = False) -> list[dict[str, Any]]:
    """Load records from a JSON list, wrapped JSON object, or JSONL file."""
    target = Path(path)
    if not target.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Record artifact does not exist: {target}")
    text = target.read_text(encoding="utf-8", errors="strict" if strict else "ignore").strip()
    if not text:
        return []

    def objects(records: list[Any]) -> list[dict[str, Any]]:
        if strict and any(not isinstance(item, dict) for item in records):
            raise ValueError(f"{target}: manifest rows must be JSON objects")
        return [item for item in records if isinstance(item, dict)]

    if text[0] in "[{":
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return objects(payload)
            if isinstance(payload, dict):
                for key in ("records", "scores", "data"):
                    seq = payload.get(key)
                    if isinstance(seq, list):
                        return objects(seq)
                return [payload]
        except json.JSONDecodeError:
            pass
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return objects(records) if strict else records


def write_json(path: str | Path, payload: Any, *, sort_keys: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=sort_keys) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    rows: Iterable[Mapping[str, Any]],
    path: str | Path,
    *,
    atomic: bool = False,
    sort_keys: bool = False,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(
        json.dumps(to_jsonable(row), ensure_ascii=False, sort_keys=sort_keys) + "\n"
        for row in rows
    )
    if atomic:
        tmp = target.with_name(f".{target.name}.tmp")
        tmp.write_text(lines, encoding="utf-8")
        tmp.replace(target)
        return
    target.write_text(lines, encoding="utf-8")


def sha256_file(path: str | Path | None) -> str | None:
    if path is None:
        return None
    target = Path(path)
    if not target.is_file():
        return None
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_shard(video_id: str, num_shards: int) -> int:
    digest = hashlib.sha256(video_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def select_manifest_shard(
    manifest: list[dict[str, Any]],
    *,
    num_shards: int,
    shard_id: int,
) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return list(manifest)
    return [item for item in manifest if stable_shard(str(item["video_id"]), num_shards) == shard_id]


def resolve_video_path(
    row: Mapping[str, Any],
    *,
    video_root: str | Path | None = None,
    surface: str | None = None,
    prefer_eval_video: bool = True,
) -> str:
    """Resolve a row's video path with optional root anchoring and continuation preference."""
    scoring_surface = str(surface or row.get("scoring_video_surface") or "").strip()
    if scoring_surface in {"full_continuation", "source_generated_concat"}:
        keys: tuple[str, ...] = ("scoring_video_path", "original_concat_path", "path", "video_path")
    elif prefer_eval_video:
        keys = ("eval_video_path", "path", "video_path")
    else:
        keys = ("path", "video_path")
    for field in keys:
        value = row.get(field)
        if not value:
            continue
        candidate = Path(str(value))
        if not candidate.is_absolute() and video_root is not None:
            candidate = Path(video_root) / candidate
        return str(candidate)
    return ""
