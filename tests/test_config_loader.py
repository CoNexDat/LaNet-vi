"""Tests for the YAML configuration loader."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from lanet_vi.io.config_loader import load_config_from_yaml
from lanet_vi.models.config import CommunityConfig


def test_removed_settings_in_old_yaml_are_ignored(tmp_path: Path):
    """A file written before 5.2 may still carry the spiral/spatial keys; they load silently."""
    path = tmp_path / "old.yaml"
    path.write_text(
        "layout:\n"
        "  use_spiral_layout: true\n"
        "  spiral_k: 3.0\n"
        "  use_spatial_hashing: false\n"
        "  seed: 7\n"
    )
    config = load_config_from_yaml(path)
    assert config.layout.seed == 7
    assert not hasattr(config.layout, "use_spiral_layout")


def test_community_colormap_must_be_a_matplotlib_colormap():
    """A typo in community.colormap is a short validation error, not matplotlib's list."""
    assert CommunityConfig(colormap="Set3").colormap == "Set3"
    with pytest.raises(ValidationError, match="not a matplotlib colormap name"):
        CommunityConfig(colormap="tabl20")


def test_removed_renderer_key_is_ignored(tmp_path: Path):
    """The dead top-level `renderer` key of older files loads without effect."""
    from lanet_vi.io.config_loader import save_config_to_yaml
    from lanet_vi.models.config import LaNetConfig

    path = tmp_path / "old.yaml"
    path.write_text("renderer: plotly\nvisualization:\n  width: 640\n")
    config = load_config_from_yaml(path)
    assert config.visualization.width == 640
    assert not hasattr(config, "renderer")

    out = tmp_path / "new.yaml"
    save_config_to_yaml(LaNetConfig(), out)
    assert "renderer" not in out.read_text()
