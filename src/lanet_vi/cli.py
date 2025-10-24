"""Command-line interface for LaNet-vi."""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from lanet_vi.core.network import Network
from lanet_vi.io.config_loader import load_config_from_yaml, save_config_to_yaml
from lanet_vi.io.writers import write_decomposition_csv, write_decomposition_json
from lanet_vi.logging_config import setup_logging
from lanet_vi.models.config import (
    BackgroundColor,
    ColorScheme,
    CommunityConfig,
    CoordDistributionAlgorithm,
    DecompositionConfig,
    DecompositionType,
    GraphConfig,
    LaNetConfig,
    LayoutConfig,
    StrengthIntervalMethod,
    VisualizationConfig,
)

app = typer.Typer(
    name="lanet-vi",
    help="Large scale network visualization using k-core and k-dense decomposition",
)
console = Console()


@app.command()
def visualize(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input edge list file"),
    output: Path = typer.Option(
        "output.png", "--output", "-o", help="Output visualization file"
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", "-c", help="YAML configuration file"
    ),
    decomp: DecompositionType = typer.Option(
        DecompositionType.KCORES, "--decomp", "-d", help="Decomposition type"
    ),
    names: Optional[Path] = typer.Option(None, "--names", help="Node names file"),
    colors_file: Optional[Path] = typer.Option(None, "--colors-file", help="Node colors file"),
    cores_file: Optional[Path] = typer.Option(
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
    edge_alpha: float = typer.Option(0.6, "--edge-alpha", help="Edge transparency (0.0-1.0)"),
    min_edge_width: float = typer.Option(0.08, "--min-edge-width", help="Minimum edge width"),
    max_edge_width: float = typer.Option(0.3, "--max-edge-width", help="Maximum edge width"),
    node_size_scale: float = typer.Option(0.5, "--node-size-scale", help="Node size multiplier"),
    node_edge_color: Optional[str] = typer.Option(
        None, "--node-edge-color", help="Node edge color"
    ),
    show_size_legend: bool = typer.Option(True, "--show-size-legend", help="Show size legend"),
    gradient_edges: bool = typer.Option(
        True, "--gradient-edges", help="Use gradient edge coloring"
    ),
    epsilon: float = typer.Option(0.40, "--epsilon", help="Controls ring overlapping"),
    delta: float = typer.Option(1.3, "--delta", help="Distance between components"),
    gamma: float = typer.Option(1.5, "--gamma", help="Component diameter"),
    font_zoom: float = typer.Option(1.0, "--font-zoom", help="Font zoom factor"),
    legend_fontsize: Optional[float] = typer.Option(
        None, "--legend-fontsize", help="Legend font size (auto-scales if not set)"
    ),
    from_layer: int = typer.Option(0, "--from-layer", help="Start from this layer"),
    granularity: int = typer.Option(-1, "--granularity", help="Groups in weighted graphs"),
    strength_intervals: StrengthIntervalMethod = typer.Option(
        StrengthIntervalMethod.EQUAL_SIZE,
        "--strength-intervals",
        help="Strength interval method",
    ),
    coord_distribution: CoordDistributionAlgorithm = typer.Option(
        CoordDistributionAlgorithm.CLASSIC,
        "--coord-distribution",
        help="Component distribution algorithm",
    ),
    alpha: float = typer.Option(1.0, "--alpha", help="Component ratio formula constant"),
    beta: float = typer.Option(1.0, "--beta", help="Component ratio formula exponent"),
    seed: int = typer.Option(0, "--seed", help="Random seed"),
    draw_circles: bool = typer.Option(False, "--draw-circles", help="Draw component borders"),
    no_cliques: bool = typer.Option(False, "--no-cliques", help="Omit cliques in central core"),
    color_scale_max: Optional[int] = typer.Option(
        None, "--color-scale-max", help="Max value for color scale"
    ),
    show_degree_scale: bool = typer.Option(
        True, "--show-degree-scale", help="Show degree scale legend"
    ),
    # Community detection options
    detect_communities: bool = typer.Option(
        False, "--detect-communities", help="Detect and visualize communities"
    ),
    community_algorithm: str = typer.Option(
        "louvain",
        "--community-algorithm",
        help="Community detection algorithm (louvain or greedy_modularity)",
    ),
    community_resolution: float = typer.Option(
        1.0,
        "--community-resolution",
        help="Resolution parameter for Louvain (higher = more communities)",
    ),
    color_by_community: bool = typer.Option(
        True, "--color-by-community", help="Color nodes by community instead of k-core"
    ),
    draw_community_boundaries: bool = typer.Option(
        True, "--draw-community-boundaries", help="Draw boundaries around communities"
    ),
    # Spiral layout options
    use_spiral_layout: bool = typer.Option(
        False, "--use-spiral-layout", help="Use spiral layout algorithm"
    ),
    spiral_k: float = typer.Option(10.0, "--spiral-K", help="Spiral scaling constant"),
    spiral_beta: float = typer.Option(1.5, "--spiral-beta", help="Spiral tightness parameter"),
    spiral_separation: float = typer.Option(
        1.0, "--spiral-separation", help="Target separation between nodes in spiral"
    ),
    # Logging options
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log to file"),
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Build configuration
        task = progress.add_task("Building configuration...", total=None)

        # Load from YAML if provided, otherwise use defaults
        if config_file:
            config = load_config_from_yaml(config_file)
            # Override with command-line arguments if provided
            if weighted:
                config.graph.weighted = weighted
            if multigraph:
                config.graph.multigraph = multigraph
        else:
            config = LaNetConfig(
                graph=GraphConfig(
                    weighted=weighted,
                    multigraph=multigraph,
                    directed=directed,
                ),
                decomposition=DecompositionConfig(
                    decomp_type=decomp,
                    from_layer=from_layer,
                    granularity=granularity,
                    strength_intervals=strength_intervals,
                    no_cliques=no_cliques,
                ),
                visualization=VisualizationConfig(
                    background=background,
                    color_scheme=color_scheme,
                    width=width,
                    height=height,
                    epsilon=epsilon,
                    delta=delta,
                    gamma=gamma,
                    font_zoom=font_zoom,
                    legend_fontsize=legend_fontsize,
                    edges_percent=edges_percent,
                    min_edges=min_edges,
                    edge_alpha=edge_alpha,
                    min_edge_width=min_edge_width,
                    max_edge_width=max_edge_width,
                    node_size_scale=node_size_scale,
                    node_edge_color=node_edge_color,
                    show_size_legend=show_size_legend,
                    gradient_edges=gradient_edges,
                    draw_circles=draw_circles,
                    show_degree_scale=show_degree_scale,
                    color_scale_max_value=color_scale_max,
                ),
                layout=LayoutConfig(
                    coord_distribution=coord_distribution,
                    alpha=alpha,
                    beta=beta,
                    seed=seed,
                    use_spiral_layout=use_spiral_layout,
                    spiral_k=spiral_k,
                    spiral_beta=spiral_beta,
                    spiral_separation=spiral_separation,
                ),
                community=CommunityConfig(
                    detect_communities=detect_communities,
                    algorithm=community_algorithm,
                    resolution=community_resolution,
                    color_by_community=color_by_community,
                    draw_boundaries=draw_community_boundaries,
                ),
            )

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
        result = network.decompose()

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
    default_config = LaNetConfig(
        decomposition=DecompositionConfig(decomp_type=decomp)
    )

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
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log to file"),
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
    p: Optional[float] = typer.Option(
        None, "--probability", "-p", help="Edge probability (Erdős-Rényi G(n,p))"
    ),
    m: Optional[int] = typer.Option(
        None, "--edges", "-e", help="Number of edges (Erdős-Rényi G(n,m) or BA/powerlaw attachment)"
    ),
    # Watts-Strogatz parameters
    k: Optional[int] = typer.Option(
        None,
        "--neighbors",
        "-k",
        help="Each node connected to k nearest neighbors (Watts-Strogatz)",
    ),
    rewire_p: Optional[float] = typer.Option(
        None, "--rewire", "-r", help="Rewiring probability (Watts-Strogatz)"
    ),
    # Powerlaw cluster parameter
    triangle_p: Optional[float] = typer.Option(
        None, "--triangle-prob", "-t", help="Triangle formation probability (powerlaw-cluster)"
    ),
    # General options
    directed: bool = typer.Option(False, "--directed", help="Generate directed graph"),
    weighted: bool = typer.Option(False, "--weighted", help="Add random edge weights"),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress console output"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Log to file"),
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
