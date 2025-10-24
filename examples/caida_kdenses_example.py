"""Example: Visualizing CAIDA AS-Relationships with K-Denses decomposition.

This example demonstrates k-denses (m-core) decomposition, which is based on
triangle density rather than node degree. K-denses identify more cohesive
subgraphs than k-cores and are useful for community detection.
"""

from pathlib import Path

from lanet_vi import DecompositionType, LaNetConfig, Network
from lanet_vi.io.readers import read_caida_snapshot
from lanet_vi.io.writers import write_decomposition_csv


def visualize_caida_kdenses():
    """
    Visualize CAIDA AS-Relationships using k-denses decomposition.

    K-denses vs K-cores:
    - K-cores: Based on node degree (number of neighbors)
    - K-denses: Based on triangle count (more cohesive structures)
    - K-denses typically produces fewer, denser shells
    - Better for identifying tightly-knit communities

    This example:
    1. Downloads the CAIDA snapshot from the web
    2. Creates a NetworkX graph
    3. Computes k-denses decomposition
    4. Generates visualization
    5. Exports decomposition results
    """
    print("=" * 70)
    print("CAIDA AS-Relationships K-Denses Visualization Example")
    print("=" * 70)

    # CAIDA snapshot URL (same as k-cores example for comparison)
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
        local_file = Path("20170101.as-rel.txt.bz2")
        if not local_file.exists():
            print(f"      ✗ Local file not found: {local_file}")
            print("\n      Please download the file manually:")
            print(f"      wget {url}")
            return

        graph, dataframe = read_caida_snapshot(local_file.as_posix())
        print(f"      ✓ Loaded from local file")

    # Configure visualization using optimized defaults
    print("\n[2/5] Configuring visualization...")
    config = LaNetConfig()

    print(f"      • Resolution: {config.visualization.width}x{config.visualization.height}")
    print(f"      • Background: {config.visualization.background}")
    print(f"      • Decomposition: K-DENSES (triangle-based)")
    print(f"      • Using optimized defaults for smooth visualization")

    # Create Network instance
    print("\n[3/5] Computing k-denses decomposition...")
    print("      Note: K-denses is slower than k-cores due to triangle counting")
    net = Network(graph, config)

    # Compute k-denses decomposition
    result = net.decompose(DecompositionType.KDENSES)

    print(f"      ✓ Decomposition complete")
    print(f"      • Min k-dense: {result.min_index}")
    print(f"      • Max k-dense: {result.max_index}")
    print(f"      • Components: {len(result.components)}")

    # Show top k-denses
    print(f"\n      Top k-denses distribution:")
    from collections import Counter
    dense_counts = Counter(result.node_indices.values())
    for k in sorted(dense_counts.keys(), reverse=True)[:5]:
        print(f"        k={k:2d}: {dense_counts[k]:5d} nodes")

    # K-denses typically produces fewer shells than k-cores
    print(f"\n      K-denses range: {result.min_index}-{result.max_index}")
    print(f"      (Compare with k-cores which typically ranges 1-79 for this dataset)")

    # Generate visualization
    print("\n[4/5] Generating visualization...")
    output_file = "caida_as_relationships_kdenses.png"
    net.visualize(output_file)
    print(f"      ✓ Saved: {output_file}")

    # Export decomposition results
    print("\n[5/5] Exporting decomposition data...")
    denses_csv = "caida_denses.csv"
    write_decomposition_csv(result, denses_csv)
    print(f"      ✓ Saved: {denses_csv}")

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
    print("K-Denses Decomposition Complete!")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  • {output_file} - Network visualization")
    print(f"  • {denses_csv} - K-denses decomposition (AS number, k-dense)")

    print(f"\nAbout K-Denses:")
    print(f"  K-denses decomposition identifies cohesive subgraphs based on")
    print(f"  triangle density rather than node degree. A node belongs to the")
    print(f"  k-dense if it participates in at least k triangles.")
    print(f"")
    print(f"  Comparison with K-cores:")
    print(f"  • K-denses: Triangle-based → identifies tight communities")
    print(f"  • K-cores: Degree-based → identifies hierarchical structure")
    print(f"")
    print(f"  The k-denses visualization shows more concentrated cores with")
    print(f"  fewer intermediate shells, highlighting the most cohesive parts")
    print(f"  of the Internet topology.")


if __name__ == "__main__":
    visualize_caida_kdenses()
