"""Tests for the YAML configuration loader."""

from pathlib import Path

from lanet_vi.io.config_loader import load_config_from_yaml


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
