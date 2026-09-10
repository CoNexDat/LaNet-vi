"""Tests for package version metadata."""

import re
from importlib.metadata import version
from pathlib import Path

import lanet_vi


def test_version_matches_installed_metadata():
    """__version__ is read from the installed package metadata."""
    assert lanet_vi.__version__ == version("lanet-vi")


def test_version_matches_pyproject():
    """pyproject.toml is the single source of truth for the version."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(), re.MULTILINE)
    assert match is not None
    assert lanet_vi.__version__ == match.group(1)
