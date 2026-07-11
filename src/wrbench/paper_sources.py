"""Download and verify the pinned Natural-25 paper TV2V source assets."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


class SourcePreparationError(RuntimeError):
    """Raised when source preparation would accept or overwrite bad state."""


DownloadFunction = Callable[[str, Path, bool], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_relpath(row: dict[str, Any]) -> Path:
    explicit = str(row.get("local_relpath") or "").strip()
    if explicit:
        return Path(explicit)
    hf_path = Path(str(row.get("hf_path") or ""))
    if hf_path.parts and hf_path.parts[0] == "videos":
        return Path(*hf_path.parts[1:])
    return Path()


def _destination(source_root: Path, row: dict[str, Any]) -> Path:
    relative = _local_relpath(row)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SourcePreparationError(
            f"{row.get('task_variant_id')}: local_relpath must be relative to --source-video-root"
        )
    destination = (source_root / relative).resolve()
    try:
        destination.relative_to(source_root)
    except ValueError as exc:
        raise SourcePreparationError(
            f"{row.get('task_variant_id')}: local_relpath escapes --source-video-root"
        ) from exc
    return destination


def _expected_asset(row: dict[str, Any]) -> tuple[str, int]:
    sha256 = str(row.get("sha256") or "").lower()
    byte_count = row.get("bytes")
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256):
        raise SourcePreparationError(f"{row.get('task_variant_id')}: invalid SHA256")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count <= 0:
        raise SourcePreparationError(f"{row.get('task_variant_id')}: bytes must be a positive integer")
    return sha256, byte_count


def _verify_asset(path: Path, row: dict[str, Any]) -> None:
    sha256, byte_count = _expected_asset(row)
    if not path.is_file():
        raise SourcePreparationError(f"{row.get('task_variant_id')}: source asset is missing: {path}")
    if path.stat().st_size != byte_count:
        raise SourcePreparationError(
            f"{row.get('task_variant_id')}: source asset byte count mismatch: {path}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != sha256:
        raise SourcePreparationError(
            f"{row.get('task_variant_id')}: source asset SHA256 {actual_sha256} != expected {sha256}: {path}"
        )


def source_download_url(row: dict[str, Any]) -> str:
    """Build an anonymous, immutable Hugging Face raw-file URL."""
    repo_id = str(row.get("hf_repo_id") or row.get("hf_repo") or "")
    revision = str(row.get("hf_revision") or row.get("hf_asset_revision") or "")
    hf_path = str(row.get("hf_path") or "")
    if not repo_id or not revision or not hf_path or revision == "main":
        raise SourcePreparationError(f"{row.get('task_variant_id')}: incomplete immutable HF source locator")
    quoted_path = urllib.parse.quote(hf_path, safe="/")
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{quoted_path}"


def _download_to_part(url: str, part_path: Path, resume: bool) -> None:
    offset = part_path.stat().st_size if resume and part_path.is_file() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request) as response:  # noqa: S310 - URL is pinned by the release manifest.
        status = int(getattr(response, "status", response.getcode()))
        append = offset > 0 and status == 206
        mode = "ab" if append else "wb"
        with part_path.open(mode) as fh:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)


def build_source_video_task_map(
    rows: list[dict[str, Any]],
    *,
    release_id: str,
) -> dict[str, Any]:
    """Create the strict task-id map consumed by ``run_natural25_generation.py``."""
    tasks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        task_variant_id = str(row.get("task_variant_id") or "")
        if not task_variant_id or task_variant_id in seen:
            raise SourcePreparationError(f"duplicate or empty task_variant_id {task_variant_id!r}")
        seen.add(task_variant_id)
        tasks.append(
            {
                "bytes": row["bytes"],
                "decoded_frame_count": row["decoded_frame_count"],
                "event_tier": row["event_tier"],
                "family_id": row["family_id"],
                "fps": row["fps"],
                "height": row["height"],
                "hf_path": row["hf_path"],
                "hf_repo_id": row.get("hf_repo_id") or row["hf_repo"],
                "hf_revision": row.get("hf_revision") or row["hf_asset_revision"],
                "sha256": row["sha256"],
                "source_catalog_variant_id": row["source_catalog_variant_id"],
                "source_video": _local_relpath(row).as_posix(),
                "source_video_id": row["source_video_id"],
                "source_video_origin": row["source_video_origin"],
                "source_video_is_repeated_first_frame": False,
                "task_variant_id": task_variant_id,
                "video_asset_id": row["video_asset_id"],
                "width": row["width"],
            }
        )
    return {
        "release_id": release_id,
        "schema_version": "wrbench.source_video_task_map.v1",
        "source_path_policy": "Every source_video path is relative to the explicit --source-video-root.",
        "tasks": tasks,
    }


def _task_map_bytes(rows: list[dict[str, Any]], *, release_id: str) -> bytes:
    payload = build_source_video_task_map(rows, release_id=release_id)
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def prepare_paper_tv2v_sources(
    rows: list[dict[str, Any]],
    *,
    release_id: str,
    source_root: str | Path,
    task_map_path: str | Path | None = None,
    mode: str,
    resume: bool = False,
    overwrite_existing: bool = False,
    downloader: DownloadFunction = _download_to_part,
) -> dict[str, Any]:
    """Plan, download, or verify pinned assets and emit a runner task map.

    ``mode='plan'`` and ``mode='verify-only'`` are side-effect free. Download
    mode refuses corrupt or conflicting existing files unless the caller passes
    ``overwrite_existing=True`` explicitly.
    """
    if mode not in {"plan", "download", "verify-only"}:
        raise ValueError(f"unsupported preparation mode {mode!r}")
    if resume and mode != "download":
        raise ValueError("resume is only valid in download mode")
    if overwrite_existing and mode != "download":
        raise ValueError("overwrite_existing is only valid in download mode")
    root = Path(source_root).expanduser().resolve()
    resolved_task_map = (
        Path(task_map_path).expanduser().resolve()
        if task_map_path is not None
        else (root / "source_video_task_map.json").resolve()
    )
    try:
        resolved_task_map.relative_to(root)
    except ValueError as exc:
        raise SourcePreparationError("task map must be written inside --source-video-root") from exc

    task_map_bytes = _task_map_bytes(rows, release_id=release_id)
    if mode == "download" and resolved_task_map.exists():
        if not resolved_task_map.is_file():
            raise SourcePreparationError(f"task map path exists and is not a file: {resolved_task_map}")
        if resolved_task_map.read_bytes() != task_map_bytes and not overwrite_existing:
            raise SourcePreparationError(
                f"refusing to overwrite conflicting task map without --overwrite-existing: {resolved_task_map}"
            )

    results: list[dict[str, Any]] = []
    for row in rows:
        destination = _destination(root, row)
        item = {
            "destination": str(destination),
            "task_variant_id": row.get("task_variant_id"),
            "url": source_download_url(row),
        }
        if mode == "plan":
            item["status"] = "would_verify" if destination.is_file() else "would_download"
            results.append(item)
            continue
        if mode == "verify-only":
            _verify_asset(destination, row)
            item["status"] = "verified"
            results.append(item)
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            try:
                _verify_asset(destination, row)
            except SourcePreparationError:
                if not overwrite_existing:
                    raise SourcePreparationError(
                        f"refusing to overwrite corrupt existing source without --overwrite-existing: {destination}"
                    )
                destination.unlink()
            else:
                item["status"] = "verified_existing"
                results.append(item)
                continue

        part_path = destination.with_name(destination.name + ".part")
        if part_path.exists() and not resume:
            if not overwrite_existing:
                raise SourcePreparationError(
                    f"partial download exists; pass --resume or --overwrite-existing: {part_path}"
                )
            part_path.unlink()
        downloader(item["url"], part_path, resume)
        _verify_asset(part_path, row)
        part_path.replace(destination)
        item["status"] = "downloaded"
        results.append(item)

    if mode == "download":
        resolved_task_map.parent.mkdir(parents=True, exist_ok=True)
        temp_map = resolved_task_map.with_name(resolved_task_map.name + ".tmp")
        temp_map.write_bytes(task_map_bytes)
        temp_map.replace(resolved_task_map)
    return {
        "mode": mode,
        "release_id": release_id,
        "rows": len(rows),
        "source_root": str(root),
        "task_map": str(resolved_task_map),
        "results": results,
    }
