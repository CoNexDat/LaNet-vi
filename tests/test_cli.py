"""Smoke tests for the Typer command-line interface."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from lanet_vi.cli import app

runner = CliRunner()


def test_help_lists_commands():
    """Top-level --help shows all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("visualize", "config", "info", "generate"):
        assert command in result.output


def test_info_reports_graph_size(small_edge_list: Path):
    """`info` prints node and edge counts for an edge list."""
    result = runner.invoke(app, ["info", str(small_edge_list)])
    assert result.exit_code == 0, result.output
    assert "7" in result.output  # nodes
    assert "8" in result.output  # edges


def test_visualize_writes_png_and_cores(small_edge_list: Path, tmp_path: Path):
    """`visualize` renders a PNG and exports the decomposition."""
    output = tmp_path / "out.png"
    cores = tmp_path / "cores.csv"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(output),
            "--cores-file",
            str(cores),
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists() and output.stat().st_size > 0
    lines = cores.read_text().strip().splitlines()
    assert lines[0] == "node_id,kcores_index"
    assert len(lines) == 8  # header + 7 nodes


def test_visualize_kdenses(small_edge_list: Path, tmp_path: Path):
    """`visualize --decomp kdenses` also produces an image."""
    output = tmp_path / "kdenses.png"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(output),
            "--decomp",
            "kdenses",
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists() and output.stat().st_size > 0


def test_generate_writes_edge_list(tmp_path: Path):
    """`generate` creates a random graph edge list."""
    output = tmp_path / "random.txt"
    result = runner.invoke(
        app,
        [
            "generate",
            "--output",
            str(output),
            "--model",
            "erdos-renyi",
            "--nodes",
            "30",
            "--probability",
            "0.2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert len(output.read_text().strip().splitlines()) > 0


def test_visualize_rejects_unknown_community_algorithm(small_edge_list: Path, tmp_path: Path):
    """An unsupported --community-algorithm is a usage error, not a crash."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(tmp_path / "x.png"),
            "--community-algorithm",
            "bogus",
        ],
    )
    assert result.exit_code == 2
    # Rich may wrap and colour the option name, so check the message text instead
    assert "Unknown community algorithm" in result.output


def test_config_yaml_round_trip(tmp_path: Path):
    """`lanet-vi config` writes YAML that `load_config_from_yaml` accepts (#23)."""
    from lanet_vi.io.config_loader import load_config_from_yaml

    cfg = tmp_path / "c.yaml"
    result = runner.invoke(app, ["config", str(cfg), "--decomp", "kdenses"])
    assert result.exit_code == 0, result.output
    assert "python/object" not in cfg.read_text()

    loaded = load_config_from_yaml(cfg)
    assert loaded.decomposition.decomp_type == "kdenses"


def test_cli_flags_override_config_file(small_edge_list: Path, tmp_path: Path):
    """Explicit flags win over --config; unset flags keep the file's values (#23)."""
    cfg = tmp_path / "c.yaml"
    cfg.write_text("visualization:\n  width: 300\n  height: 300\n  edge_alpha: 0.25\n")
    cores = tmp_path / "cores.csv"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--config",
            str(cfg),
            "--decomp",
            "kdenses",
            "--output",
            str(tmp_path / "o.png"),
            "--cores-file",
            str(cores),
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    # --decomp from the command line was honoured despite --config
    assert cores.read_text().splitlines()[0] == "node_id,kdenses_index"


def test_build_config_precedence():
    """_build_config applies only explicitly given parameters on top of the YAML/defaults."""
    from lanet_vi.cli import _build_config

    result = runner.invoke(app, ["visualize", "--help"])
    assert result.exit_code == 0

    class FakeCtx:
        params = {"width": 1000, "height": 500, "edge_alpha": 0.9, "decomp": "kdenses"}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                pass

            src = Src()
            src.name = "COMMANDLINE" if name in ("width", "height") else "DEFAULT"
            return src

    config = _build_config(FakeCtx(), None)  # type: ignore[arg-type]
    assert (config.visualization.width, config.visualization.height) == (1000, 500)
    assert config.visualization.edge_alpha == 0.6  # default kept
    assert config.decomposition.decomp_type == "kcores"  # not explicit


def test_boolean_flags_have_no_forms(small_edge_list: Path, tmp_path: Path):
    """Default-true booleans can be switched off (#23)."""
    from unittest.mock import patch

    from lanet_vi.core.network import Network

    seen = {}
    original = Network.visualize

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN202
        seen["cfg"] = self.config.visualization
        return original(self, *args, **kwargs)

    with patch.object(Network, "visualize", spy):
        result = runner.invoke(
            app,
            [
                "visualize",
                "--input",
                str(small_edge_list),
                "--output",
                str(tmp_path / "o.png"),
                "--no-show-degree-scale",
                "--no-show-color-legend",
                "--no-gradient-edges",
                "--width",
                "300",
                "--height",
                "300",
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    cfg = seen["cfg"]
    assert (cfg.show_degree_scale, cfg.show_color_legend, cfg.gradient_edges) == (
        False,
        False,
        False,
    )


def test_names_file_enables_labels(small_edge_list: Path, tmp_path: Path):
    """--names turns node labels on unless --no-node-labels is given (#23)."""
    from unittest.mock import patch

    names = tmp_path / "names.txt"
    names.write_text("0 zero\n1 one\n")
    seen: dict[str, bool] = {}

    from lanet_vi.core.network import Network

    original = Network.visualize

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN202
        seen["labels"] = self.config.visualization.show_node_labels
        return original(self, *args, **kwargs)

    with patch.object(Network, "visualize", spy):
        result = runner.invoke(
            app,
            [
                "visualize",
                "--input",
                str(small_edge_list),
                "--names",
                str(names),
                "--output",
                str(tmp_path / "o.png"),
                "--width",
                "300",
                "--height",
                "300",
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    assert seen["labels"] is True


def test_build_config_validates_after_merge(tmp_path: Path):
    """A width/height pair that is invalid in the YAML alone passes once flags fix it."""
    from lanet_vi.cli import _build_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text("visualization:\n  width: 3200\n  height: 800\n")  # aspect 4.0: invalid

    class FakeCtx:
        params = {"height": 1600}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                name = "COMMANDLINE"

            return Src()

    config = _build_config(FakeCtx(), cfg)  # type: ignore[arg-type]
    assert (config.visualization.width, config.visualization.height) == (3200, 1600)


def test_deprecated_show_size_legend_alias():
    """show_size_legend: false folds into show_degree_scale; an explicit flag wins over it."""
    from lanet_vi.cli import _build_config
    from lanet_vi.models.config import VisualizationConfig

    folded = VisualizationConfig(show_size_legend=False)
    assert folded.show_degree_scale is False and folded.show_size_legend is True

    class FakeCtx:
        params = {"show_degree_scale": True}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                name = "COMMANDLINE"

            return Src()

    import tempfile

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "c.yaml"
        cfg.write_text("visualization:\n  show_size_legend: false\n")
        config = _build_config(FakeCtx(), cfg)  # type: ignore[arg-type]
    assert config.visualization.show_degree_scale is True


def test_build_config_reports_malformed_section_via_validation(tmp_path: Path):
    """A non-mapping section in the YAML is reported by validation, not a TypeError."""
    from pydantic import ValidationError

    from lanet_vi.cli import _build_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text("visualization: null\n")

    class FakeCtx:
        params = {"width": 1000}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                name = "COMMANDLINE"

            return Src()

    with pytest.raises(ValidationError):
        _build_config(FakeCtx(), cfg)  # type: ignore[arg-type]


def test_yaml_show_node_labels_survives_without_names(small_edge_list: Path, tmp_path: Path):
    """visualization.show_node_labels: true in YAML is kept when --names is omitted."""
    from unittest.mock import patch

    from lanet_vi.core.network import Network

    cfg = tmp_path / "c.yaml"
    cfg.write_text("visualization:\n  show_node_labels: true\n  width: 300\n  height: 300\n")
    seen = {}
    original = Network.visualize

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN202
        seen["labels"] = self.config.visualization.show_node_labels
        return original(self, *args, **kwargs)

    with patch.object(Network, "visualize", spy):
        result = runner.invoke(
            app,
            [
                "visualize",
                "--input",
                str(small_edge_list),
                "--config",
                str(cfg),
                "--output",
                str(tmp_path / "o.png"),
                "--quiet",
            ],
        )
    assert result.exit_code == 0, result.output
    assert seen["labels"] is True
