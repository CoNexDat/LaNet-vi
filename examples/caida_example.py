"""Example: Visualizing CAIDA AS-Relationships data with LaNet-vi."""

from pathlib import Path

from lanet_vi import DecompositionType, LaNetConfig, Network
from lanet_vi.io.readers import read_caida_snapshot
from lanet_vi.io.writers import write_decomposition_csv


def visualize_caida_snapshot():
    """
    Download and visualize CAIDA AS-Relationships data.

    This example:
    1. Downloads the CAIDA snapshot from the web
    2. Creates a NetworkX graph
    3. Computes k-core decomposition
    4. Generates visualization
    5. Exports decomposition results
    """
    print("=" * 70)
    print("CAIDA AS-Relationships Visualization Example")
    print("=" * 70)

    # CAIDA snapshot URL
    # url = "https://publicdata.caida.org/datasets/as-relationships/serial-1/20251001.as-rel.txt.bz2"
    url = "https://publicdata.caida.org/datasets/as-relationships/serial-1/20170101.as-rel.txt.bz2"

    print(f"\n[1/5] Downloading CAIDA snapshot...")
    print(f"      URL: {url}")

    try:
        # Download and parse CAIDA data
        graph, dataframe = read_caida_snapshot(url, timeout=60)

        print(f"      ✓ Downloaded successfully")
        print(f"      • AS nodes: {graph.number_of_nodes()}")
        print(f"      • Relationships: {graph.number_of_edges()}")
        print(f"      • DataFrame shape: {dataframe.shape}")

    except Exception as e:
        print(f"      ✗ Failed to download: {e}")
        print("\n      Using local file instead...")

        # Fallback: try local file if download fails
        local_file = Path("20251001.as-rel.txt.bz2")
        if not local_file.exists():
            print(f"      ✗ Local file not found: {local_file}")
            print("\n      Please download the file manually:")
            print(f"      wget {url}")
            return

        graph, dataframe = read_caida_snapshot(local_file.as_posix())
        print(f"      ✓ Loaded from local file")

    # Configure visualization using LaNet-vi optimized defaults
    # All parameters have been tuned based on CAIDA dataset and are now the defaults
    print("\n[2/5] Configuring visualization...")
    config = LaNetConfig()

    print(f"      • Resolution: {config.visualization.width}x{config.visualization.height}")
    print(f"      • Background: {config.visualization.background}")
    print(f"      • Edge visibility: {config.visualization.edges_percent*100}% (min: {config.visualization.min_edges})")
    print(f"      • Epsilon: {config.visualization.epsilon} (radial spread)")
    print(f"      • Using optimized defaults for smooth, readable visualization")

    # Create Network instance
    print("\n[3/5] Computing k-core decomposition...")
    net = Network(graph, config)

    # Compute k-core decomposition
    result = net.decompose(DecompositionType.KCORES)

    print(f"      ✓ Decomposition complete")
    print(f"      • Min k-core: {result.min_index}")
    print(f"      • Max k-core: {result.max_index}")
    print(f"      • Components: {len(result.components)}")

    # Show top k-cores
    print(f"\n      Top k-cores distribution:")
    from collections import Counter
    core_counts = Counter(result.node_indices.values())
    for k in sorted(core_counts.keys(), reverse=True)[:5]:
        print(f"        k={k:2d}: {core_counts[k]:5d} nodes")

    # Generate visualization
    print("\n[4/5] Generating visualization...")
    output_file = "caida_as_relationships_kcores.png"
    net.visualize(output_file)
    print(f"      ✓ Saved: {output_file}")

    # Export decomposition results
    print("\n[5/5] Exporting decomposition data...")
    cores_csv = "caida_cores.csv"
    write_decomposition_csv(result, cores_csv)
    print(f"      ✓ Saved: {cores_csv}")

    # Display network statistics
    print("\n" + "=" * 70)
    print("Network Statistics")
    print("=" * 70)
    metadata = net.get_metadata()
    print(f"  Autonomous Systems (nodes): {metadata['num_nodes']:,}")
    print(f"  AS relationships (edges):   {metadata['num_edges']:,}")
    print(f"  Average degree:              {metadata['avg_degree']:.2f}")
    print(f"  Network density:             {metadata['density']:.6f}")
    print(f"  Max degree (hub AS):         {metadata['max_degree']}")
    print(f"  Min degree:                  {metadata['min_degree']}")

    print("\n" + "=" * 70)
    print("Visualization complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  • {output_file} - Network visualization")
    print(f"  • {cores_csv} - K-core decomposition (AS number, k-core)")
    print(f"\nThe visualization shows the hierarchical structure of the")
    print(f"Internet's AS-level topology, with central ASes in inner shells")
    print(f"and peripheral networks in outer shells.")


def visualize_from_local_file():
    """
    Alternative: Load from local .bz2 file.

    Use this if you've already downloaded the file:
    wget https://publicdata.caida.org/datasets/as-relationships/serial-1/20251001.as-rel.txt.bz2
    """
    print("Loading from local file: 20251001.as-rel.txt.bz2")

    from lanet_vi.io.readers import read_edge_list

    # Read the edge list directly (format is source|target|relationship_type)
    # The relationship types are: -1=customer-provider, 0=peer-peer, 1=sibling-sibling

    config = LaNetConfig()
    config.graph.directed = False  # Treat as undirected for visualization

    # Read the bz2 file
    graph = read_edge_list(
        "20251001.as-rel.txt.bz2",
        delimiter="|",
        comment="#",
        weighted=False,
        directed=False,
    )

    print(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    # Create network and visualize
    net = Network(graph, config)
    net.decompose(DecompositionType.KCORES)
    net.visualize("caida_local.png")

    print("Saved: caida_local.png")


if __name__ == "__main__":
    # Try downloading from web first
    visualize_caida_snapshot()

    # Uncomment to use local file instead:
    # visualize_from_local_file()
