"""Tests for the random graph generators behind ``lanet-vi generate``."""

import networkx as nx
import pytest

from lanet_vi.generators import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_powerlaw_cluster,
    generate_watts_strogatz,
)


def test_erdos_renyi_gnp_is_undirected_and_reproducible():
    """G(n, p) has n nodes, is undirected, and the seed fixes the edge set."""
    first = generate_erdos_renyi(50, p=0.1, seed=1)
    second = generate_erdos_renyi(50, p=0.1, seed=1)
    assert first.number_of_nodes() == 50
    assert not first.is_directed()
    assert set(first.edges()) == set(second.edges())
    assert set(first.edges()) != set(generate_erdos_renyi(50, p=0.1, seed=2).edges())


def test_erdos_renyi_gnm_has_exactly_m_edges():
    """G(n, m) has exactly m edges."""
    G = generate_erdos_renyi(30, m=45, seed=0)
    assert G.number_of_nodes() == 30
    assert G.number_of_edges() == 45


def test_erdos_renyi_directed_variants():
    """``directed=True`` yields a DiGraph under both models."""
    assert generate_erdos_renyi(20, p=0.2, seed=0, directed=True).is_directed()
    G = generate_erdos_renyi(20, m=30, seed=0, directed=True)
    assert G.is_directed()
    assert G.number_of_edges() == 30


def test_erdos_renyi_requires_exactly_one_of_p_and_m():
    """Neither or both of p and m is a usage error."""
    with pytest.raises(ValueError, match="Either p or m"):
        generate_erdos_renyi(10)
    with pytest.raises(ValueError, match="both p and m"):
        generate_erdos_renyi(10, p=0.1, m=5)


def test_barabasi_albert_edge_count_and_reproducibility():
    """BA(n, m) has (n - m) * m edges (the first m nodes start as a star) and a fixed seed."""
    G = generate_barabasi_albert(40, 3, seed=7)
    assert G.number_of_nodes() == 40
    assert G.number_of_edges() == (40 - 3) * 3
    assert nx.is_connected(G)
    assert set(G.edges()) == set(generate_barabasi_albert(40, 3, seed=7).edges())


def test_watts_strogatz_keeps_the_ring_edge_count():
    """Rewiring preserves the n * k / 2 edges of the ring lattice."""
    G = generate_watts_strogatz(30, 4, 0.3, seed=3)
    assert G.number_of_nodes() == 30
    assert G.number_of_edges() == 30 * 4 // 2
    assert set(G.edges()) == set(generate_watts_strogatz(30, 4, 0.3, seed=3).edges())


def test_watts_strogatz_without_rewiring_is_the_ring_lattice():
    """With p = 0 every node keeps exactly k neighbors."""
    G = generate_watts_strogatz(20, 4, 0.0)
    assert set(dict(G.degree()).values()) == {4}


def test_powerlaw_cluster_is_more_clustered_than_barabasi_albert():
    """Holme-Kim adds triangles: with p = 1 its clustering exceeds plain BA on the same seed."""
    G = generate_powerlaw_cluster(200, 2, 1.0, seed=5)
    assert G.number_of_nodes() == 200
    ba = generate_barabasi_albert(200, 2, seed=5)
    assert nx.average_clustering(G) > nx.average_clustering(ba)
    assert set(G.edges()) == set(generate_powerlaw_cluster(200, 2, 1.0, seed=5).edges())
