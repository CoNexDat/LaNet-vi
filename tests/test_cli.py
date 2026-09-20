"""Smoke tests for the Typer command-line interface."""

import re
from pathlib import Path

import networkx as nx
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
    # Rich may wrap and color the option name, so check the message text instead
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
    cfg.write_text("visualization:\n  width: 300\n  height: 300\n  opacity: 0.25\n")
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
    # --decomp from the command line was honored despite --config
    assert cores.read_text().splitlines()[0] == "node_id,kdenses_index"


def test_build_config_precedence():
    """_build_config applies only explicitly given parameters on top of the YAML/defaults."""
    from lanet_vi.cli import _build_config

    result = runner.invoke(app, ["visualize", "--help"])
    assert result.exit_code == 0

    class FakeCtx:
        params = {"width": 1000, "height": 500, "opacity": 0.9, "decomp": "kdenses"}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                pass

            src = Src()
            src.name = "COMMANDLINE" if name in ("width", "height") else "DEFAULT"
            return src

    config = _build_config(FakeCtx(), None)  # type: ignore[arg-type]
    assert (config.visualization.width, config.visualization.height) == (1000, 500)
    assert config.visualization.opacity == 0.2  # default kept
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
    """A YAML that is invalid on its own (custom intervals, no file) passes once a flag fixes it."""
    from lanet_vi.cli import _build_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text("decomposition:\n  strength_intervals: custom\n")  # needs a file: invalid
    boundaries = tmp_path / "b.txt"
    boundaries.write_text("0\n1\n2\n")

    class FakeCtx:
        params = {"strength_intervals_file": boundaries}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                name = "COMMANDLINE"

            return Src()

    config = _build_config(FakeCtx(), cfg)  # type: ignore[arg-type]
    assert config.decomposition.strength_intervals == "custom"
    assert config.decomposition.strength_intervals_file == boundaries


def test_any_aspect_ratio_is_accepted(small_edge_list: Path, tmp_path: Path):
    """Wide pictures such as 3200x800 are rendered at exactly that size (#24)."""
    from PIL import Image

    out = tmp_path / "wide.png"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--width",
            "640",
            "--height",
            "160",
            "--output",
            str(out),
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    with Image.open(out) as image:
        assert image.size == (640, 160)


def test_deprecated_edge_alpha_folds_into_opacity(tmp_path: Path):
    """edge_alpha in a YAML file or --edge-alpha sets opacity unless opacity is given too."""
    from lanet_vi.cli import _build_config
    from lanet_vi.models.config import VisualizationConfig

    assert VisualizationConfig(edge_alpha=0.7).opacity == 0.7
    assert VisualizationConfig(edge_alpha=0.7, opacity=0.4).opacity == 0.4

    cfg = tmp_path / "c.yaml"
    cfg.write_text("visualization:\n  edge_alpha: 0.55\n")

    class FakeCtx:
        params = {"edge_alpha": 0.9}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                pass

            src = Src()
            src.name = "COMMANDLINE" if name == "edge_alpha" else "DEFAULT"
            return src

    assert _build_config(FakeCtx(), cfg).visualization.opacity == 0.9  # type: ignore[arg-type]
    FakeCtx.params = {}
    assert _build_config(FakeCtx(), cfg).visualization.opacity == 0.55  # type: ignore[arg-type]

    # Both flags explicit: the current one wins, whatever the order they are merged in
    class BothCtx:
        params = {"opacity": 0.5, "edge_alpha": 0.9}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                name = "COMMANDLINE"

            return Src()

    assert _build_config(BothCtx(), None).visualization.opacity == 0.5  # type: ignore[arg-type]


def test_config_template_has_no_deprecated_aliases(tmp_path: Path):
    """lanet-vi config omits edge_alpha/show_size_legend, so editing the template works."""
    import yaml

    from lanet_vi.io.config_loader import load_config_from_yaml

    cfg = tmp_path / "c.yaml"
    result = runner.invoke(app, ["config", str(cfg)])
    assert result.exit_code == 0, result.output
    visualization = yaml.safe_load(cfg.read_text())["visualization"]
    assert "edge_alpha" not in visualization and "show_size_legend" not in visualization
    assert visualization["opacity"] == 0.2

    # A template where the user replaces opacity by the old name still applies it
    visualization.pop("opacity")
    visualization["edge_alpha"] = 0.9
    cfg.write_text(yaml.safe_dump({"visualization": visualization}))
    assert load_config_from_yaml(cfg).visualization.opacity == 0.9


def test_deprecated_show_size_legend_alias():
    """show_size_legend: false folds into show_degree_scale; an explicit flag wins over it."""
    from lanet_vi.cli import _build_config
    from lanet_vi.models.config import VisualizationConfig

    folded = VisualizationConfig(show_size_legend=False)
    assert folded.show_degree_scale is False and folded.show_size_legend is True
    # The alias never overrides an explicit current field
    explicit = VisualizationConfig(show_degree_scale=True, show_size_legend=False)
    assert explicit.show_degree_scale is True

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

    class NoFlags:
        params: dict = {}

        def get_parameter_source(self, name: str):  # noqa: D102
            return None

    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d) / "c.yaml"
        cfg.write_text("visualization:\n  show_degree_scale: true\n  show_size_legend: false\n")
        config = _build_config(NoFlags(), cfg)  # type: ignore[arg-type]
    assert config.visualization.show_degree_scale is True  # current field wins over alias


def test_build_config_reports_malformed_section_via_validation(tmp_path: Path):
    """A non-mapping section in the YAML is reported as a usage error, not a TypeError."""
    import typer

    from lanet_vi.cli import _build_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text("visualization: null\n")

    class FakeCtx:
        params = {"width": 1000}

        def get_parameter_source(self, name: str):  # noqa: D102
            class Src:
                name = "COMMANDLINE"

            return Src()

    with pytest.raises(typer.BadParameter, match="visualization"):
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


def test_no_node_labels_overrides_names(small_edge_list: Path, tmp_path: Path):
    """--no-node-labels keeps labels off even when --names is given."""
    from unittest.mock import patch

    from lanet_vi.core.network import Network

    names = tmp_path / "names.txt"
    names.write_text("0 zero\n1 one\n")
    seen: dict[str, bool] = {}
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
                "--no-node-labels",
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
    assert seen["labels"] is False


def test_weighted_strength_flags_reach_the_decomposition(small_edge_list: Path, tmp_path: Path):
    """--maximum-strength and --strength-intervals custom/--strength-intervals-file (#20)."""
    from unittest.mock import patch

    from lanet_vi.core.network import Network

    seen: dict[str, object] = {}
    original = Network.visualize

    def spy(self, *args, **kwargs):  # noqa: ANN001, ANN202
        seen["decomposition"] = self.config.decomposition
        seen["p_function"] = self.decomposition.p_function
        return original(self, *args, **kwargs)

    intervals = tmp_path / "intervals.txt"
    intervals.write_text("1\n2\n")
    common = [
        "visualize",
        "--input",
        str(small_edge_list),
        "--output",
        str(tmp_path / "o.png"),
        "--weighted",
        "--width",
        "300",
        "--height",
        "300",
        "--quiet",
    ]

    with patch.object(Network, "visualize", spy):
        result = runner.invoke(app, [*common, "--granularity", "2", "--maximum-strength", "6"])
    assert result.exit_code == 0, result.output
    assert seen["decomposition"].maximum_strength == 6.0
    assert seen["p_function"] == [0.0, 3.0, 6.0]

    with patch.object(Network, "visualize", spy):
        custom = ["--strength-intervals", "custom", "--strength-intervals-file", str(intervals)]
        result = runner.invoke(app, [*common, *custom])
    assert result.exit_code == 0, result.output
    assert seen["decomposition"].strength_intervals_file == intervals
    assert seen["p_function"] == [0.0, 1.0, 2.0]

    result = runner.invoke(app, [*common, "--strength-intervals", "custom"])
    assert result.exit_code != 0
    assert "strength_intervals_file" in result.output


def test_config_file_with_non_mapping_top_level_is_a_usage_error(
    small_edge_list: Path, tmp_path: Path
):
    """A list at the top level or malformed YAML is reported by the CLI, not a traceback."""
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- 1\n- 2\n")
    result = _invoke_with_config(small_edge_list, tmp_path, cfg)
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)

    cfg.write_text("visualization: [unterminated\n")
    result = _invoke_with_config(small_edge_list, tmp_path, cfg)
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)

    result = _invoke_with_config(small_edge_list, tmp_path, tmp_path / "missing.yaml")
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)


def _invoke_with_config(small_edge_list: Path, tmp_path: Path, cfg: Path):  # noqa: ANN202
    return runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(tmp_path / "o.png"),
            "--config",
            str(cfg),
            "--quiet",
        ],
    )


def test_unusable_custom_intervals_file_is_a_usage_error(small_edge_list: Path, tmp_path: Path):
    """A missing or malformed --strength-intervals-file fails before the network loads."""
    for path, content in ((tmp_path / "missing.txt", None), (tmp_path / "bad.txt", "3\n1\n")):
        if content is not None:
            path.write_text(content)
        result = runner.invoke(
            app,
            [
                "visualize",
                "--input",
                str(small_edge_list),
                "--output",
                str(tmp_path / "o.png"),
                "--weighted",
                "--strength-intervals",
                "custom",
                "--strength-intervals-file",
                str(path),
                "--quiet",
            ],
        )
        assert result.exit_code == 2, result.output
        assert isinstance(result.exception, SystemExit)


def test_decomposition_value_errors_are_usage_errors(small_edge_list: Path, tmp_path: Path):
    """--decomp dcores on an undirected graph is reported by the CLI, not a traceback."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(tmp_path / "o.png"),
            "--decomp",
            "dcores",
            "--quiet",
        ],
    )
    assert result.exit_code == 2, result.output
    assert isinstance(result.exception, SystemExit)


def test_window_flag_and_yaml_round_trip(small_edge_list: Path, tmp_path: Path):
    """--window takes four fractions; the template writes it as a list that loads back."""
    import yaml

    from lanet_vi.io.config_loader import load_config_from_yaml

    out = tmp_path / "crop.png"
    result = runner.invoke(
        app,
        [
            "visualize",
            "-i",
            str(small_edge_list),
            "-o",
            str(out),
            "--width",
            "120",
            "--height",
            "100",
            "--window",
            "0",
            "0.5",
            "0",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out.exists() and out.stat().st_size > 0

    result = runner.invoke(
        app,
        ["visualize", "-i", str(small_edge_list), "-o", str(out), "--window", "1", "0", "0", "1"],
    )
    assert result.exit_code != 0 and "window" in result.output

    cfg = tmp_path / "c.yaml"
    assert runner.invoke(app, ["config", str(cfg)]).exit_code == 0
    data = yaml.safe_load(cfg.read_text())
    assert data["visualization"]["window"] == [0.0, 1.0, 0.0, 1.0]
    data["visualization"]["window"] = [0.25, 0.75, 0.0, 1.0]
    cfg.write_text(yaml.safe_dump(data))
    assert load_config_from_yaml(cfg).visualization.window == (0.25, 0.75, 0.0, 1.0)


def test_visualize_from_layer_draws_the_central_subgraph(small_edge_list: Path, tmp_path: Path):
    """`--from-layer 2` keeps the two triangles (index 2) and drops the pendant node."""
    output = tmp_path / "core.png"
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
            "--from-layer",
            "2",
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists() and output.stat().st_size > 0
    rows = [line.split(",") for line in cores.read_text().strip().splitlines()[1:]]
    assert {int(node) for node, _ in rows} == {0, 1, 2, 3, 4, 5}  # node 6 is the pendant
    assert {int(index) for _, index in rows} == {2}


def test_visualize_from_layer_beyond_max_is_a_usage_error(small_edge_list: Path, tmp_path: Path):
    """A layer above the maximum index exits with a usage error, not a traceback."""
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(tmp_path / "x.png"),
            "--from-layer",
            "9",
            "--quiet",
        ],
    )
    assert result.exit_code == 2, result.output
    assert "from_layer=9 leaves no node" in result.output


def test_visualize_detect_communities_reports_and_draws_them(
    small_edge_list: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`--detect-communities` runs the detection, reports it and reaches the renderer."""
    from lanet_vi.visualization import matplotlib_renderer as mr

    calls: list[str] = []
    monkeypatch.setattr(mr, "draw_community_boundaries", lambda *a, **k: calls.append("hull"))
    output = tmp_path / "communities.png"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(output),
            "--detect-communities",
            "--community-algorithm",
            "greedy_modularity",
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists() and output.stat().st_size > 0
    assert "Communities (greedy_modularity): 2" in result.output
    assert calls == ["hull"]

    # --no-draw-community-boundaries keeps the detection but drops the hulls
    calls.clear()
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(small_edge_list),
            "--output",
            str(output),
            "--detect-communities",
            "--no-draw-community-boundaries",
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Communities (louvain)" in result.output
    assert calls == []

    # Without the flag nothing is detected or reported
    result = runner.invoke(
        app,
        ["visualize", "--input", str(small_edge_list), "--output", str(output), "--quiet"],
    )
    assert result.exit_code == 0, result.output
    assert "Communities" not in result.output


def test_dcore_table_flag_writes_the_table_and_needs_dcores(tmp_path: Path):
    """--dcore-table writes the (k, l)-core table; with another decomposition it is an error."""
    from lanet_vi.decomposition.dcores import compute_dcore_table

    edges = tmp_path / "digraph.txt"
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 2), (1, 3)])
    edges.write_text("".join(f"{u} {v}\n" for u, v in graph.edges()))
    table = tmp_path / "dcores_list.txt"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(edges),
            "--output",
            str(tmp_path / "d.png"),
            "--directed",
            "--decomp",
            "dcores",
            "--dcore-table",
            str(table),
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [tuple(map(int, line.split())) for line in table.read_text().splitlines()[1:]]
    expected = compute_dcore_table(graph)
    assert {(node, k, out_min) for node, k, out_min in rows} == {
        (node, k, out_min) for out_min, row in expected.items() for node, k in row.items()
    }
    assert "(k, l)-core table" in result.output

    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(edges),
            "--output",
            str(tmp_path / "k.png"),
            "--dcore-table",
            str(tmp_path / "no.txt"),
            "--quiet",
        ],
    )
    assert result.exit_code == 2
    # Rich may wrap and color the option names, so check the message text instead
    assert "directed graph" in result.output
    assert not (tmp_path / "no.txt").exists()


def _plain(output: str) -> str:
    """Strip the ANSI escapes Rich emits on a color terminal (CI runs with one)."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", output)


def test_kconn_flag_reports_and_writes_the_kconnectivity(tmp_path: Path):
    """--kconn prints the summary; --kconn-file writes the C++ kconn.log table."""
    edges = tmp_path / "clique.txt"
    graph = nx.complete_graph(5)
    graph.add_edge(0, 5)
    edges.write_text("".join(f"{u} {v}\n" for u, v in graph.edges()))
    kconn = tmp_path / "kconn.txt"
    result = runner.invoke(
        app,
        [
            "visualize",
            "--input",
            str(edges),
            "--output",
            str(tmp_path / "k.png"),
            "--kconn",
            "--kconn-type",
            "strict",
            "--kconn-file",
            str(kconn),
            "--color-scheme",
            "bw",
            "--width",
            "300",
            "--height",
            "300",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "K-connectivity (strict): 6 of 6 nodes are k-connected" in _plain(result.output)
    lines = kconn.read_text().splitlines()
    assert lines[0] == "# node shell_index k_connectivity"
    assert lines[1] == "5 1 1"
    assert sorted(lines[2:]) == [f"{v} 4 4" for v in range(5)]
    assert (tmp_path / "k.png").exists()


def test_kconn_usage_errors(small_edge_list: Path, tmp_path: Path):
    """--kconn-file without --kconn, and --kconn with k-denses or weights, exit with 2."""
    base = ["visualize", "--input", str(small_edge_list), "--output", str(tmp_path / "x.png")]
    result = runner.invoke(app, [*base, "--kconn-file", str(tmp_path / "k.txt"), "--quiet"])
    assert result.exit_code == 2, result.output
    assert "--kconn-file needs --kconn" in _plain(result.output)

    result = runner.invoke(app, [*base, "--kconn", "--decomp", "kdenses", "--quiet"])
    assert result.exit_code == 2, result.output
    assert "k-core" in result.output

    result = runner.invoke(app, [*base, "--kconn", "--weighted", "--quiet"])
    assert result.exit_code == 2, result.output
    assert "weighted" in result.output
