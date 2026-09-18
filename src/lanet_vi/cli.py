"""Command-line interface for LaNet-vi."""

import logging
from enum import Enum
from pathlib import Path
from typing import Any

import typer
import yaml
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lanet_vi.core.network import Network
from lanet_vi.decomposition.kcores import read_custom_intervals
from lanet_vi.io.config_loader import read_config_yaml, save_config_to_yaml
from lanet_vi.io.writers import write_decomposition_csv, write_decomposition_json
from lanet_vi.logging_config import setup_logging
from lanet_vi.models.config import (
    DEPRECATED_ALIASES,
    BackgroundColor,
    ColorScheme,
    CoordDistributionAlgorithm,
    DecompositionConfig,
    DecompositionType,
    GraphConfig,
    LaNetConfig,
    MeasureType,
    StrengthIntervalMethod,
)

# CLI parameter name -> (config section, field). Only parameters the user gave
# explicitly override the YAML file / defaults (C++ precedence: defaults <
# config file < command line).
_CLI_TO_CONFIG: dict[str, tuple[str, str]] = {
    "weighted": ("graph", "weighted"),
    "multigraph": ("graph", "multigraph"),
    "directed": ("graph", "directed"),
    "decomp": ("decomposition", "decomp_type"),
    "measure": ("decomposition", "measure"),
    "from_layer": ("decomposition", "from_layer"),
    "granularity": ("decomposition", "granularity"),
    "strength_intervals": ("decomposition", "strength_intervals"),
    "maximum_strength": ("decomposition", "maximum_strength"),
    "strength_intervals_file": ("decomposition", "strength_intervals_file"),
    "no_cliques": ("decomposition", "no_cliques"),
    "background": ("visualization", "background"),
    "color_scheme": ("visualization", "color_scheme"),
    "width": ("visualization", "width"),
    "height": ("visualization", "height"),
    "epsilon": ("visualization", "epsilon"),
    "delta": ("visualization", "delta"),
    "gamma": ("visualization", "gamma"),
    "font_zoom": ("visualization", "font_zoom"),
    "legend_fontsize": ("visualization", "legend_fontsize"),
    "edges_percent": ("visualization", "edges_percent"),
    "min_edges": ("visualization", "min_edges"),
    "opacity": ("visualization", "opacity"),
    "edge_alpha": ("visualization", "edge_alpha"),
    "node_size_scale": ("visualization", "node_size_scale"),
    "node_edge_color": ("visualization", "node_edge_color"),
    "show_size_legend": ("visualization", "show_size_legend"),
    "gradient_edges": ("visualization", "gradient_edges"),
    "draw_circles": ("visualization", "draw_circles"),
    "show_degree_scale": ("visualization", "show_degree_scale"),
    "show_color_legend": ("visualization", "show_color_legend"),
    "show_node_labels": ("visualization", "show_node_labels"),
    "color_scale_max": ("visualization", "color_scale_max_value"),
    "coord_distribution": ("layout", "coord_distribution"),
    "alpha": ("layout", "alpha"),
    "beta": ("layout", "beta"),
    "ratio_constant": ("layout", "ratio_constant"),
    "seed": ("layout", "seed"),
    "use_spiral_layout": ("layout", "use_spiral_layout"),
    "spiral_k": ("layout", "spiral_k"),
    "spiral_beta": ("layout", "spiral_beta"),
    "spiral_separation": ("layout", "spiral_separation"),
    "detect_communities": ("community", "detect_communities"),
    "community_algorithm": ("community", "algorithm"),
    "community_resolution": ("community", "resolution"),
    "color_by_community": ("community", "color_by_community"),
    "draw_community_boundaries": ("community", "draw_boundaries"),
}

# Names of click's ParameterSource members that mean "the user set this". Compared by
# name because Typer >= 0.20 vendors click and does not re-export the enum.
_EXPLICIT_SOURCES = ("COMMANDLINE", "ENVIRONMENT")


def _given_explicitly(ctx: typer.Context, param: str) -> bool:
    """Return True if ``param`` came from the command line or the environment."""
    source = ctx.get_parameter_source(param)
    return source is not None and source.name in _EXPLICIT_SOURCES


def _build_config(ctx: typer.Context, config_file: Path | None) -> LaNetConfig:
    """Merge defaults, the YAML file and explicitly given CLI flags; validate once.

    Precedence is the C++ one: defaults < config file < command line. Validation
    runs only on the merged values, so a flag can fix a value that would be
    invalid on its own in the file (for example one side of the aspect ratio).
    """
    data: dict[str, Any] = LaNetConfig().model_dump(mode="json")

    if config_file is not None:
        try:
            file_data = read_config_yaml(config_file)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--config") from exc
        for section, values in file_data.items():
            if isinstance(values, dict) and isinstance(data.get(section), dict):
                if section == "visualization":
                    # Deprecated aliases: honor them only when the current field is absent
                    values = dict(values)
                    for alias, field in DEPRECATED_ALIASES.items():
                        if alias in values:
                            values.setdefault(field, values.pop(alias))
                data[section].update(values)
            else:
                data[section] = values

    for param, (section, field) in _CLI_TO_CONFIG.items():
        if param not in ctx.params or not _given_explicitly(ctx, param):
            continue
        if not isinstance(data.get(section), dict):
            # Malformed section in the file (e.g. `visualization: null`): leave it for
            # model validation to report instead of failing here with a TypeError
            continue
        value = ctx.params[param]
        data[section][field] = value.value if isinstance(value, Enum) else value
        if field in DEPRECATED_ALIASES:
            # Deprecated CLI aliases (--show-size-legend, --edge-alpha): the current flag
            # wins when both are given explicitly
            target = DEPRECATED_ALIASES[field]
            target_param = next(
                (p for p, (s, f) in _CLI_TO_CONFIG.items() if (s, f) == (section, target)), None
            )
            if target_param is None or not _given_explicitly(ctx, target_param):
                data[section][target] = value

    try:
        config = LaNetConfig.model_validate(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise typer.BadParameter(f"Invalid configuration: {problems}") from exc

    if config.decomposition.strength_intervals == StrengthIntervalMethod.CUSTOM:
        # Fail before loading the network if the boundaries file is unusable
        try:
            read_custom_intervals(config.decomposition.strength_intervals_file)
        except (OSError, ValueError) as exc:
            raise typer.BadParameter(str(exc), param_hint="--strength-intervals-file") from exc
    return config


app = typer.Typer(
    name="lanet-vi",
    help="Large scale network visualization using k-core and k-dense decomposition",
)
console = Console()


@app.command()
def visualize(
    ctx: typer.Context,
    input_file: Path = typer.Option(..., "--input", "-i", help="Input edge list file"),
    output: Path = typer.Option("output.png", "--output", "-o", help="Output visualization file"),
    config_file: Path | None = typer.Option(None, "--config", "-c", help="YAML configuration file"),
    decomp: DecompositionType = typer.Option(
        DecompositionType.KCORES, "--decomp", "-d", help="Decomposition type"
    ),
    measure: MeasureType = typer.Option(
        MeasureType.MCORE,
        "--measure",
        help="Centrality measure named in the k-dense legend: m-core (k-dense minus 2) or k-dense",
    ),
    names: Path | None = typer.Option(None, "--names", help="Node names file"),
    colors_file: Path | None = typer.Option(None, "--colors-file", help="Node colors file"),
    cores_file: Path | None = typer.Option(
        None, "--cores-file", help="Export decomposition to file"
    ),
    weighted: bool = typer.Option(False, "--weighted", "-w", help="Graph has edge weights"),
    multigraph: bool = typer.Option(False, "--multigraph", help="Allow repeated edges"),
    directed: bool = typer.Option(
        False, "--directed", help="Graph is directed (required for dcores)"
    ),
    width: int = typer.Option(2400, "--width", "-W", help="Image width in pixels"),
    height: int = typer.Option(2400, "--height", "-H", help="Image height in pixels"),
    background: BackgroundColor = typer.Option(
        BackgroundColor.BLACK, "--background", help="Background color"
    ),
    color_scheme: ColorScheme = typer.Option(
        ColorScheme.COLOR, "--color-scheme", help="Color scheme"
    ),
    edges_percent: float = typer.Option(
        0.5, "--edges-percent", help="Percent of visible edges (0.0-1.0)"
    ),
    min_edges: int = typer.Option(50000, "--min-edges", help="Minimum number of visible edges"),
    opacity: float = typer.Option(0.2, "--opacity", help="Edge opacity (0.0-1.0)"),
    edge_alpha: float = typer.Option(0.2, "--edge-alpha", hidden=True),
    min_edge_width: float | None = typer.Option(
        None,
        "--min-edge-width",
        help="Deprecated, no effect: edge width follows the C++ degree radius (#24)",
        hidden=True,
    ),
    max_edge_width: float | None = typer.Option(
        None,
        "--max-edge-width",
        help="Deprecated, no effect: edge width follows the C++ degree radius (#24)",
        hidden=True,
    ),
    node_size_scale: float = typer.Option(
        1.0, "--node-size-scale", help="Multiplier on the node radius (C++ size at 1.0)"
    ),
    node_edge_color: str | None = typer.Option(None, "--node-edge-color", help="Node edge color"),
    show_size_legend: bool = typer.Option(
        True, "--show-size-legend/--no-show-size-legend", hidden=True
    ),
    gradient_edges: bool = typer.Option(
        True, "--gradient-edges/--no-gradient-edges", help="Use gradient edge coloring"
    ),
    epsilon: float = typer.Option(
        0.18, "--epsilon", help="Ring thickness as a fraction of its radius (C++: 0.18)"
    ),
    delta: float = typer.Option(
        1.3, "--delta", help="Shrink factor of sibling components (formula 5)"
    ),
    gamma: float = typer.Option(
        1.5, "--gamma", help="Component diameter (scales the whole picture)"
    ),
    font_zoom: float = typer.Option(1.0, "--font-zoom", help="Font zoom factor"),
    legend_fontsize: float | None = typer.Option(
        None, "--legend-fontsize", help="Legend font size (auto-scales if not set)"
    ),
    from_layer: int = typer.Option(
        0, "--from-layer", help="Start from this layer (not implemented yet, #23)"
    ),
    granularity: int = typer.Option(
        -1, "--granularity", help="Groups in weighted graphs (-1: maximum degree)"
    ),
    strength_intervals: StrengthIntervalMethod = typer.Option(
        StrengthIntervalMethod.EQUAL_SIZE,
        "--strength-intervals",
        help="How to build the strength intervals of weighted graphs",
    ),
    maximum_strength: float | None = typer.Option(
        None,
        "--maximum-strength",
        help="Upper limit of the strength intervals (to compare pictures of different networks)",
    ),
    strength_intervals_file: Path | None = typer.Option(
        None,
        "--strength-intervals-file",
        help="Interval boundaries, one per line, for --strength-intervals custom",
    ),
    coord_distribution: CoordDistributionAlgorithm = typer.Option(
        CoordDistributionAlgorithm.CLASSIC,
        "--coord-distribution",
        help="Component placement: classic rings, or pow/log circle packing of siblings",
    ),
    alpha: float = typer.Option(0.3, "--alpha", help="Circle-packing area constant (pow/log)"),
    beta: float = typer.Option(1.0, "--beta", help="Circle-packing area exponent (pow/log)"),
    ratio_constant: float | None = typer.Option(
        None, "--ratio-constant", help="Node radius factor of pow/log (default: auto-adjusted)"
    ),
    seed: int = typer.Option(0, "--seed", help="Random seed of the layout"),
    draw_circles: bool = typer.Option(False, "--draw-circles", help="Draw component borders"),
    no_cliques: bool = typer.Option(
        False, "--no-cliques", help="Spread the top core uniformly instead of by cliques"
    ),
    color_scale_max: int | None = typer.Option(
        None, "--color-scale-max", help="Max value for color scale"
    ),
    show_degree_scale: bool = typer.Option(
        True, "--show-degree-scale/--no-show-degree-scale", help="Show the degree (size) legend"
    ),
    show_color_legend: bool = typer.Option(
        True, "--show-color-legend/--no-show-color-legend", help="Show the color legend"
    ),
    show_node_labels: bool | None = typer.Option(
        None,
        "--node-labels/--no-node-labels",
        help="Draw node names (default: on when --names is given)",
    ),
    # Community detection options
    detect_communities: bool = typer.Option(
        False,
        "--detect-communities",
        help="Detect and visualize communities (not wired into rendering yet, #23)",
    ),
    community_algorithm: str = typer.Option(
        "louvain",
        "--community-algorithm",
        help="Community detection algorithm: louvain or greedy_modularity (not wired yet, #23)",
    ),
    community_resolution: float = typer.Option(
        1.0,
        "--community-resolution",
        help="Resolution parameter for Louvain (not wired yet, #23)",
    ),
    color_by_community: bool = typer.Option(
        True,
        "--color-by-community/--no-color-by-community",
        help="Color nodes by community instead of k-core (not wired into rendering yet, #23)",
    ),
    draw_community_boundaries: bool = typer.Option(
        True,
        "--draw-community-boundaries/--no-draw-community-boundaries",
        help="Draw boundaries around communities (not wired into rendering yet, #23)",
    ),
    # Spiral layout options
    use_spiral_layout: bool = typer.Option(
        False, "--use-spiral-layout", help="Use spiral layout algorithm (not implemented, #18)"
    ),
    spiral_k: float = typer.Option(
        10.0, "--spiral-K", help="Spiral scaling constant (not implemented, #18)"
    ),
    spiral_beta: float = typer.Option(
        1.5, "--spiral-beta", help="Spiral tightness parameter (not implemented, #18)"
    ),
    spiral_separation: float = typer.Option(
        1.0, "--spiral-separation", help="Target separation in spiral (not implemented, #18)"
    ),
    # Logging options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
    log_file: Path | None = typer.Option(None, "--log-file", help="Log to file"),
) -> None:
    """
    Visualize a network using k-core or k-dense decomposition.

    This command loads a network from an edge list file, computes the
    decomposition, and generates a visualization.

    Examples
    --------
        lanet-vi visualize --input network.txt --output viz.png

        lanet-vi visualize --input network.txt --decomp kdenses --weighted
    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level, log_file=log_file, quiet=quiet)

    if community_algorithm not in ("louvain", "greedy_modularity"):
        raise typer.BadParameter(
            f"Unknown community algorithm {community_algorithm!r}; "
            "use 'louvain' or 'greedy_modularity'",
            param_hint="--community-algorithm",
        )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Build configuration
        task = progress.add_task("Building configuration...", total=None)

        config = _build_config(ctx, config_file)
        if show_node_labels is None and names is not None:
            # --names implies labels unless the user said otherwise (YAML value kept
            # when no names file is given)
            config.visualization.show_node_labels = True
        decomp = DecompositionType(config.decomposition.decomp_type)

        progress.update(task, description="Loading network...")
        network = Network.from_edge_list(input_file, config)

        # Load optional data
        if names:
            progress.update(task, description="Loading node names...")
            network.load_node_names(names)

        if colors_file:
            progress.update(task, description="Loading node colors...")
            network.load_node_colors(colors_file)

        # Decompose
        progress.update(task, description=f"Computing {decomp.value} decomposition...")
        try:
            result = network.decompose()
        except ValueError as exc:
            # Input/option combinations the decomposition refuses (negative weights,
            # d-cores on an undirected graph, ...): a usage error, not a traceback
            raise typer.BadParameter(str(exc)) from exc

        console.print(
            f"[green]✓[/green] Decomposition complete: "
            f"{result.min_index} - {result.max_index}, "
            f"{len(result.components)} components"
        )

        # Export decomposition if requested
        if cores_file:
            progress.update(task, description="Exporting decomposition...")
            if cores_file.suffix == ".json":
                write_decomposition_json(result, cores_file)
            else:
                write_decomposition_csv(result, cores_file)
            console.print(f"[green]✓[/green] Exported to {cores_file}")

        # Generate visualization
        progress.update(task, description="Computing layout...")
        layout = network.compute_layout()

        progress.update(task, description="Rendering visualization...")
        network.visualize(output, layout)

        progress.update(task, description="Done!", completed=True)

    console.print(f"[green]✓[/green] Visualization saved to {output}")

    # Show metadata
    metadata = network.get_metadata()
    console.print(
        f"\n[bold]Network Statistics:[/bold]\n"
        f"  Nodes: {metadata['num_nodes']}\n"
        f"  Edges: {metadata['num_edges']}\n"
        f"  Avg Degree: {metadata['avg_degree']:.2f}\n"
        f"  Density: {metadata['density']:.4f}"
    )


@app.command()
def config(
    output: Path = typer.Argument(..., help="Output YAML configuration file"),
    decomp: DecompositionType = typer.Option(
        DecompositionType.KCORES, "--decomp", "-d", help="Decomposition type"
    ),
) -> None:
    """
    Generate a default configuration file.

    This creates a YAML file with all available configuration options
    and their default values. You can then edit this file and use it
    with the --config option in the visualize command.

    Examples
    --------
        lanet-vi config my_config.yaml

        lanet-vi config kdense_config.yaml --decomp kdenses
    """
    # Create default configuration
    default_config = LaNetConfig(decomposition=DecompositionConfig(decomp_type=decomp))

    # Save to file
    save_config_to_yaml(default_config, output)

    console.print(f"[green]✓[/green] Configuration saved to {output}")
    console.print("\nEdit this file to customize your visualization settings,")
    console.print(f"then use: [bold]lanet-vi visualize --input data.txt --config {output}[/bold]")


@app.command()
def info(
    input_file: Path = typer.Argument(..., help="Input edge list file"),
    weighted: bool = typer.Option(False, "--weighted", "-w", help="Graph has edge weights"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
    log_file: Path | None = typer.Option(None, "--log-file", help="Log to file"),
) -> None:
    """Display information about a network file."""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level, log_file=log_file, quiet=quiet)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading network...", total=None)

        config = LaNetConfig(graph=GraphConfig(weighted=weighted))
        network = Network.from_edge_list(input_file, config)

        progress.update(task, description="Done!", completed=True)

    metadata = network.get_metadata()
    console.print(f"\n[bold]Network: {input_file}[/bold]")
    console.print(f"  Nodes: {metadata['num_nodes']}")
    console.print(f"  Edges: {metadata['num_edges']}")
    console.print(f"  Max Degree: {metadata['max_degree']}")
    console.print(f"  Min Degree: {metadata['min_degree']}")
    console.print(f"  Avg Degree: {metadata['avg_degree']:.2f}")
    console.print(f"  Density: {metadata['density']:.4f}")
    console.print(f"  Directed: {metadata['is_directed']}")


@app.command()
def generate(
    output: Path = typer.Option(..., "--output", "-o", help="Output edge list file"),
    model: str = typer.Option(
        "erdos-renyi",
        "--model",
        "-m",
        help="Graph model (erdos-renyi, barabasi-albert, watts-strogatz, powerlaw-cluster)",
    ),
    n: int = typer.Option(..., "--nodes", "-n", help="Number of nodes"),
    # Erdős-Rényi parameters
    p: float | None = typer.Option(
        None, "--probability", "-p", help="Edge probability (Erdős-Rényi G(n,p))"
    ),
    m: int | None = typer.Option(
        None, "--edges", "-e", help="Number of edges (Erdős-Rényi G(n,m) or BA/powerlaw attachment)"
    ),
    # Watts-Strogatz parameters
    k: int | None = typer.Option(
        None,
        "--neighbors",
        "-k",
        help="Each node connected to k nearest neighbors (Watts-Strogatz)",
    ),
    rewire_p: float | None = typer.Option(
        None, "--rewire", "-r", help="Rewiring probability (Watts-Strogatz)"
    ),
    # Powerlaw cluster parameter
    triangle_p: float | None = typer.Option(
        None, "--triangle-prob", "-t", help="Triangle formation probability (powerlaw-cluster)"
    ),
    # General options
    directed: bool = typer.Option(False, "--directed", help="Generate directed graph"),
    weighted: bool = typer.Option(False, "--weighted", help="Add random edge weights"),
    seed: int | None = typer.Option(None, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
    log_file: Path | None = typer.Option(None, "--log-file", help="Log to file"),
) -> None:
    r"""
    Generate random graphs for testing and demonstration.

    Examples
    --------
        # Erdős-Rényi with edge probability
        lanet-vi generate --output er.txt --model erdos-renyi --nodes 1000 --probability 0.01

        # Erdős-Rényi with fixed number of edges
        lanet-vi generate --output er.txt --model erdos-renyi --nodes 1000 --edges 5000

        # Barabási-Albert scale-free network
        lanet-vi generate --output ba.txt --model barabasi-albert --nodes 1000 --edges 3

        # Watts-Strogatz small-world network
        lanet-vi generate --output ws.txt --model watts-strogatz --nodes 1000 \\
            --neighbors 6 --rewire 0.3

        # Powerlaw cluster graph
        lanet-vi generate --output pc.txt --model powerlaw-cluster --nodes 1000 \\
            --edges 3 --triangle-prob 0.5
    """
    from lanet_vi.generators import (
        generate_barabasi_albert,
        generate_erdos_renyi,
        generate_powerlaw_cluster,
        generate_watts_strogatz,
    )
    from lanet_vi.io.writers import write_edge_list

    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level, log_file=log_file, quiet=quiet)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Generating {model} graph...", total=None)

        # Generate graph based on model
        if model == "erdos-renyi":
            if p is None and m is None:
                console.print(
                    "[red]Error:[/red] Erdős-Rényi requires either --probability or --edges"
                )
                raise typer.Exit(1)
            graph = generate_erdos_renyi(n=n, p=p, m=m, seed=seed, directed=directed)

        elif model == "barabasi-albert":
            if m is None:
                console.print(
                    "[red]Error:[/red] Barabási-Albert requires --edges (attachment count)"
                )
                raise typer.Exit(1)
            graph = generate_barabasi_albert(n=n, m=m, seed=seed)

        elif model == "watts-strogatz":
            if k is None or rewire_p is None:
                console.print("[red]Error:[/red] Watts-Strogatz requires --neighbors and --rewire")
                raise typer.Exit(1)
            graph = generate_watts_strogatz(n=n, k=k, p=rewire_p, seed=seed)

        elif model == "powerlaw-cluster":
            if m is None or triangle_p is None:
                console.print(
                    "[red]Error:[/red] Powerlaw cluster requires --edges and --triangle-prob"
                )
                raise typer.Exit(1)
            graph = generate_powerlaw_cluster(n=n, m=m, p=triangle_p, seed=seed)

        else:
            console.print(f"[red]Error:[/red] Unknown model: {model}")
            console.print(
                "Choose from: erdos-renyi, barabasi-albert, watts-strogatz, powerlaw-cluster"
            )
            raise typer.Exit(1)

        # Add random weights if requested
        if weighted:
            import random

            if seed is not None:
                random.seed(seed)
            for u, v in graph.edges():
                graph[u][v]["weight"] = random.uniform(0.1, 10.0)

        progress.update(task, description="Writing to file...", completed=False)

        # Write to file
        write_edge_list(graph, output, include_weights=weighted)

        progress.update(task, description="Done!", completed=True)

    console.print(f"\n[green]✓[/green] Generated {model} graph:")
    console.print(f"  Nodes: {graph.number_of_nodes()}")
    console.print(f"  Edges: {graph.number_of_edges()}")
    console.print(f"  Directed: {graph.is_directed()}")
    console.print(f"  Output: {output}")


def main() -> None:
    """Run the CLI application."""
    app()


if __name__ == "__main__":
    main()
