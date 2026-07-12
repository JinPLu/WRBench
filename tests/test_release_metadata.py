"""Release metadata must identify one package version and one exact license."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import wrbench
from wrbench.cli import main


ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_APACHE_2_0_SHA256 = (
    "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
)


def _toml_section(text: str, name: str) -> str:
    marker = f"[{name}]"
    section = text.split(marker, 1)[1]
    return section.split("\n[", 1)[0]


def test_package_version_has_one_build_metadata_owner() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = _toml_section(pyproject, "project")
    setuptools_dynamic = _toml_section(pyproject, "tool.setuptools.dynamic")

    assert wrbench.__version__ == "0.1.3"
    assert 'dynamic = ["version"]' in project
    assert "\nversion =" not in project
    assert 'version = { attr = "wrbench.__version__" }' in setuptools_dynamic


def test_cli_version_uses_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"wrbench {wrbench.__version__}\n"


def test_license_is_exact_official_apache_2_0_text() -> None:
    license_bytes = (ROOT / "LICENSE").read_bytes()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project = _toml_section(pyproject, "project")

    assert hashlib.sha256(license_bytes).hexdigest() == OFFICIAL_APACHE_2_0_SHA256
    assert 'license = "Apache-2.0"' in project
    assert 'license-files = ["LICENSE"]' in project
