#!/usr/bin/env bash
# Open-source verification gate for WRBench.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "== select python =="
PY=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    ver="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [ "$major" -gt 3 ] || { [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; }; then
      PY="$candidate"
      break
    fi
  fi
done
if [ -z "$PY" ]; then
  echo "FAIL: need Python >=3.10 (python3.12/python3.11/python3.10 not found)"
  exit 1
fi
echo "Using $PY ($("$PY" --version))"

echo "== leak scan =="
"$PY" - <<'PY'
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
SCAN_TARGETS = (
    "src",
    "docs",
    "examples",
    "scripts",
    ".github",
    "wrbench.runtime.example.json",
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
)

EXCLUDED_FILES = {
    "scripts/oss_verify.sh",
}

SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".venv",
    "node_modules",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp", ".tiff",
    ".pdf", ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z",
    ".npy", ".npz", ".pt", ".pth", ".ckpt", ".safetensors", ".bin",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
}

PRIVATE_RE = re.compile(
    r"(/media/datasets|10\.40\.[0-9]+|OminiEWM_Data|/tmp/ljp/|OoVMetric|"
    r"4DWorldEvolution|workplace/|gpu_proof|qwen35_vlm|P25D3D4|docs/teamwork/)"
)

LEGACY_JARGON_RE = re.compile(
    r"(iclr2026_conference|build_wrbench_vnext_main_table|P25D3D4|P22D5D6|PROMPT_MODE_P9)"
)

WRBENCHLIB_FORM_RE = re.compile(
    r"WRBenchLib/|WRBenchLib\.git|github\.com/\S*WRBenchLib|git@\S*WRBenchLib|"
    r"(?<![A-Za-z0-9_])wrbenchlib(?![A-Za-z0-9_])|"
    r"pip\s+install[^\r\n]*WRBenchLib|from\s+WRBenchLib|import\s+WRBenchLib"
)

WRBENCHLIB_ALLOWLIST_SUBSTRINGS = (
    "WRBenchLib, the generation-provenance layer in WRBench",
    "The WRBenchLib layer maps each test record",
)


def iter_files(targets: tuple[str, ...] = SCAN_TARGETS) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for target in targets:
        path = ROOT / target
        if not path.exists():
            continue
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            if rel not in EXCLUDED_FILES:
                found.append((rel, path))
            continue
        for dirpath, dirnames, filenames in os.walk(path):
            rel_dir = Path(dirpath).relative_to(ROOT).as_posix()
            if rel_dir == "docs/teamwork" or rel_dir.startswith("docs/teamwork/"):
                dirnames[:] = []
                continue
            dirnames[:] = sorted(
                d for d in dirnames
                if d not in SKIP_DIR_NAMES
                and not d.endswith(".egg-info")
                and f"{rel_dir}/{d}" != "docs/teamwork"
            )
            for fname in sorted(filenames):
                if fname == ".DS_Store":
                    continue
                fpath = Path(dirpath) / fname
                rel = fpath.relative_to(ROOT).as_posix()
                if rel in EXCLUDED_FILES or rel.startswith("docs/teamwork/"):
                    continue
                found.append((rel, fpath))
    return sorted(found, key=lambda item: item[0])


files = iter_files()
if not files:
    print("FAIL: leak scan found zero files to inspect", file=sys.stderr)
    sys.exit(1)

private_hits: list[str] = []
legacy_jargon_hits: list[str] = []
wrbenchlib_form_hits: list[str] = []
wrbenchlib_bad_hits: list[str] = []
scanned_text_files = 0

for rel, fpath in files:
    if PRIVATE_RE.search(rel):
        private_hits.append(f"{rel}:0: <filename match>")
    if LEGACY_JARGON_RE.search(rel):
        legacy_jargon_hits.append(f"{rel}:0: <filename match>")
    if WRBENCHLIB_FORM_RE.search(rel):
        wrbenchlib_form_hits.append(f"{rel}:0: <filename match>")
    elif "WRBenchLib" in rel:
        wrbenchlib_bad_hits.append(f"{rel}:0: <filename match>")

    if fpath.suffix.lower() in BINARY_EXTENSIONS:
        continue

    raw = fpath.read_bytes()
    if b"\x00" in raw:
        continue

    scanned_text_files += 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")

    for lineno, line in enumerate(text.splitlines(), start=1):
        if PRIVATE_RE.search(line):
            private_hits.append(f"{rel}:{lineno}:{line}")
        if LEGACY_JARGON_RE.search(line):
            legacy_jargon_hits.append(f"{rel}:{lineno}:{line}")
        if WRBENCHLIB_FORM_RE.search(line):
            wrbenchlib_form_hits.append(f"{rel}:{lineno}:{line}")
        if "WRBenchLib" in line:
            allowed = rel == "docs/index.html" and any(
                sub in line for sub in WRBENCHLIB_ALLOWLIST_SUBSTRINGS
            )
            if not allowed:
                wrbenchlib_bad_hits.append(f"{rel}:{lineno}:{line}")

# Also guard tests/ against legacy jargon regressions.
for rel, fpath in iter_files(("tests",)):
    if LEGACY_JARGON_RE.search(rel):
        legacy_jargon_hits.append(f"{rel}:0: <filename match>")
    if fpath.suffix.lower() in BINARY_EXTENSIONS:
        continue
    raw = fpath.read_bytes()
    if b"\x00" in raw:
        continue
    text = raw.decode("utf-8", errors="replace")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if LEGACY_JARGON_RE.search(line):
            legacy_jargon_hits.append(f"{rel}:{lineno}:{line}")

if scanned_text_files == 0:
    print("FAIL: leak scan inspected zero text files", file=sys.stderr)
    sys.exit(1)

if private_hits:
    for hit in private_hits:
        print(hit)
    print("FAIL: internal path or legacy reference found in publish tree")
    sys.exit(1)
print(f"OK: no private-path matches ({scanned_text_files} text files, {len(files)} total paths scanned)")

if legacy_jargon_hits:
    for hit in legacy_jargon_hits:
        print(hit)
    print("FAIL: legacy internal jargon or dead submission artifact found in repository")
    sys.exit(1)
print("OK: no legacy jargon or dead submission artifacts found")

if wrbenchlib_form_hits:
    for hit in wrbenchlib_form_hits:
        print(hit)
    print("FAIL: internal WRBenchLib path, package, import, or repo reference found")
    sys.exit(1)

if wrbenchlib_bad_hits:
    for hit in wrbenchlib_bad_hits:
        print(hit)
    print("FAIL: WRBenchLib is allowed only as bounded project-page provenance prose")
    sys.exit(1)
print("OK: WRBenchLib references are allowlisted")
PY

echo "== install + pytest =="
TMP="$(mktemp -d)"
"$PY" -m venv "$TMP/venv"
# shellcheck disable=SC1091
source "$TMP/venv/bin/activate"
pip install -q -U pip setuptools wheel
if ! pip install -q -e ".[dev]"; then
  echo "FAIL: editable install -e \".[dev]\" failed"
  exit 1
fi
python -m pytest -q
python - <<'PY'
import json
from importlib.metadata import version

import wrbench
from wrbench.datasets import (
    NATURAL25_PAPER_RELEASE_ID,
    NATURAL25_RELEASE_CORE_FILES,
    load_jsonl,
    load_natural25_families,
    natural25_families_path,
    natural25_first_frame_path,
    natural25_first_frames_dir,
    natural25_first_frames_manifest_path,
    natural25_release_dir,
    natural25_release_path,
    natural25_t2v_event_tails_path,
    natural25_variants_path,
    published_t2v_results_json,
)
from wrbench.eval.runtime import contract_path
from wrbench.release_validation import validate_natural25_release
from wrbench.t2v import validate_subject_anchored_prompt

families = load_natural25_families()
variants = list(load_jsonl(natural25_variants_path()))
t2v_event_tails = {row["variant_id"]: row["t2v_event_tail"] for row in load_jsonl(natural25_t2v_event_tails_path())}
manifest = json.loads(natural25_first_frames_manifest_path().read_text(encoding="utf-8"))
frames = sorted(natural25_first_frames_dir().glob("*.png"))
missing = [family_id for family_id in families if not natural25_first_frame_path(family_id).is_file()]
manifest_ids = {row["family_id"] for row in manifest}

assert len(families) == 25, len(families)
assert len(variants) == 400, len(variants)
assert len(t2v_event_tails) == 400, len(t2v_event_tails)
assert len(manifest) == 25, len(manifest)
assert len(frames) == 25, len(frames)
assert not missing, missing
assert manifest_ids == set(families), sorted(set(families) ^ manifest_ids)
assert published_t2v_results_json().is_file(), published_t2v_results_json()
bad_t2v_tails = [
    row["variant_id"]
    for row in variants
    if row["oov_gap"] == "none"
    and not validate_subject_anchored_prompt(f"Scene. background. {t2v_event_tails[row['variant_id']]}")
]
assert not bad_t2v_tails, bad_t2v_tails[:5]
paper_release = validate_natural25_release(NATURAL25_PAPER_RELEASE_ID)
assert wrbench.__version__ == version("wrbench"), wrbench.__version__
assert paper_release["status"] == "ok", paper_release
assert paper_release["paper_video_rows"] == 9600, paper_release
assert all(natural25_release_path(path).is_file() for path in NATURAL25_RELEASE_CORE_FILES)

print(
    "models", len(wrbench.list_models()),
    "contract", contract_path(),
    "natural25", natural25_families_path(),
    "first_frames", len(frames),
    "variants", len(variants),
    "paper_release", natural25_release_dir(),
    "paper_sources", paper_release["source_rows"]["rows"],
)
PY
deactivate
rm -rf "$TMP"
echo "OK: clean venv install + tests"

echo "All OSS checks passed."
