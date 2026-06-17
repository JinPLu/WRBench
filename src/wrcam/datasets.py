"""Bundled WRBench dataset paths shipped with the WRCam package."""

from __future__ import annotations

import csv
import json
from importlib import resources
from pathlib import Path
from typing import Any, Iterator


def _package_data_dir() -> Path:
    return Path(str(resources.files("wrcam") / "data"))


def data_dir() -> Path:
    return _package_data_dir()


def natural25_dir() -> Path:
    return data_dir() / "natural25"


def natural25_families_path() -> Path:
    return natural25_dir() / "families.jsonl"


def natural25_scene_events_path() -> Path:
    return natural25_dir() / "scene_events_25x4.csv"


def published_results_dir() -> Path:
    return data_dir() / "results"


def published_results_csv() -> Path:
    return published_results_dir() / "wrbench_23model_results.csv"


def published_results_json() -> Path:
    return published_results_dir() / "wrbench_23model_results.json"


def load_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_natural25_families() -> dict[str, dict[str, Any]]:
    """Load Natural-25 family records keyed by ``family_id``."""
    return {row["family_id"]: row for row in load_jsonl(natural25_families_path())}


def build_natural25_candidates(*, offscreen_area: str = "empty floor space") -> list[dict[str, Any]]:
    """Build deterministic task candidates from ``scene_events_25x4.csv``."""
    path = natural25_scene_events_path()
    candidates: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            candidates.append(
                {
                    "candidate_id": row["family_id"],
                    "offscreen_area": offscreen_area,
                    "events": {
                        "t0": row["event_T0"],
                        "t1": row["event_T1"],
                        "div_a_state_only": row["event_T2_div_a"],
                        "div_b": row["event_T2_div_b"],
                    },
                }
            )
    return candidates
