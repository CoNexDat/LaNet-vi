"""Smoke tests for the Typer command-line interface."""

from pathlib import Path

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
