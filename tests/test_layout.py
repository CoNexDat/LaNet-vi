"""Tests for the circle packing of sibling components (visualization/layout.py)."""

import numpy as np
import pytest

from lanet_vi.visualization.layout import distribute_components, packing_radii


def _no_overlap(x: np.ndarray, y: np.ndarray, r: np.ndarray) -> bool:
    n = len(x)
    return all(
        np.hypot(x[i] - x[j], y[i] - y[j]) >= r[i] + r[j] - 1e-9
        for i in range(n)
        for j in range(i + 1, n)
    )


def test_packing_radii_follow_the_area_law():
    """Radii are R sqrt(alpha w^beta), with log(1 + w) in log mode."""
    w = np.array([0.25, 0.75])
    assert packing_radii(2.0, w, 0.3, 1.0, False) == pytest.approx(2.0 * np.sqrt(0.3 * w))
    assert packing_radii(2.0, w, 0.3, 2.0, False) == pytest.approx(2.0 * np.sqrt(0.3 * w**2))
    assert packing_radii(2.0, w, 0.3, 1.0, True) == pytest.approx(2.0 * np.sqrt(0.3 * np.log1p(w)))


def test_discs_stay_inside_the_container_and_grow_until_they_touch():
    """Packed discs do not leave the container, and the packing inflates the initial radii."""
    weights = np.array([5.0, 3.0, 2.0, 1.0, 1.0, 1.0])
    x, y, r = distribute_components(1.0, -2.0, 4.0, weights, 0.3, 1.0, False, seed=0)

    assert (np.hypot(x - 1.0, y + 2.0) <= 4.0 - r + 1e-9).all()
    initial = packing_radii(4.0, weights / weights.sum(), 0.3, 1.0, False)
    assert (r > initial).all()  # alpha grew before the first failure
    # The last round is undone by 1.1: the radii are 1.01^k / 1.1 times the initial ones
    ratio = r / initial
    assert ratio == pytest.approx(np.full(len(weights), ratio[0]))
    assert (r**2).sum() < 16.0  # total area below the container's


def test_packing_is_deterministic_per_seed_and_weights():
    """The C++ re-seeded on every call; the same inputs give the same packing."""
    weights = np.array([2.0, 1.0, 1.0])
    first = distribute_components(0.0, 0.0, 1.0, weights, 0.3, 1.0, False, seed=7)
    second = distribute_components(0.0, 0.0, 1.0, weights, 0.3, 1.0, False, seed=7)
    other = distribute_components(0.0, 0.0, 1.0, weights, 0.3, 1.0, False, seed=8)
    for a, b in zip(first, second, strict=True):
        assert a == pytest.approx(b)
    assert not all(np.allclose(a, b) for a, b in zip(first, other, strict=True))


def test_heavier_discs_are_larger_and_the_result_is_free_of_overlaps_when_sparse():
    """With plenty of room the discs never overlap; radii follow the weights."""
    weights = np.array([9.0, 4.0, 1.0])
    x, y, r = distribute_components(0.0, 0.0, 10.0, weights, 0.05, 1.0, False, seed=1)
    assert r[0] > r[1] > r[2]
    assert _no_overlap(x, y, r)


def test_fewer_than_two_discs_are_centered_without_packing():
    """A lone disc gets the container center and its base radius (the C++ never packs one)."""
    x, y, r = distribute_components(3.0, 4.0, 2.0, np.array([7.0]), 0.3, 1.0, False, seed=0)
    assert (x[0], y[0]) == (3.0, 4.0)
    assert r[0] == pytest.approx(2.0 * np.sqrt(0.3))
    x, y, r = distribute_components(0.0, 0.0, 1.0, np.array([]), 0.3, 1.0, False, seed=0)
    assert len(x) == len(y) == len(r) == 0
