"""Circle packing of sibling components (the C++ ``distribute_components.cpp``).

Used by the ``pow`` / ``log`` coordinate distributions of :mod:`lanet_layout`: the child
components of a component are packed as non-overlapping discs inside its disc, each with
an area that grows with its weight, and the discs are inflated step by step until they no
longer fit.
"""

from __future__ import annotations

import math

import numpy as np

TWO_PI = 2.0 * math.pi
INCREMENT = 1.01  # growth of alpha per successful round
FACTOR_CORRECTOR = 1.10  # the final radii use the "last alpha that worked"


def packing_radii(
    radius: float, weights: np.ndarray, alpha: float, beta: float, log_mode: bool
) -> np.ndarray:
    """Radii of the discs for normalized weights: ``R sqrt(alpha w^beta)``.

    In ``log`` mode the weight enters as ``log(1 + w)``.
    """
    base = np.log1p(weights) if log_mode else weights
    return np.asarray(radius * np.sqrt(alpha * np.power(base, beta)), dtype=float)


def _random_point(
    rng: np.random.Generator, x0: float, y0: float, radius: float, r: float
) -> tuple[float, float]:
    """Draw a random point at distance ``[0, R - r)`` from the center, as the C++ did."""
    theta = rng.random() * TWO_PI
    rad = rng.random() * (radius - r)
    return x0 + rad * math.cos(theta), y0 + rad * math.sin(theta)


def _overlaps(i: int, xi: float, yi: float, x: np.ndarray, y: np.ndarray, r: np.ndarray) -> bool:
    """Whether a disc ``i`` at ``(xi, yi)`` overlaps any other disc."""
    distance = np.hypot(x - xi, y - yi)
    touching = distance < r[i] + r
    touching[i] = False
    return bool(touching.any())


def _give_new_random_position(
    c: int,
    x: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    x0: float,
    y0: float,
    radius: float,
    max_tries: int,
    rng: np.random.Generator,
) -> bool:
    """Move disc ``c`` to a random spot that overlaps no other disc (``give_new_random_position``).

    A try only counts when it overlaps; the C++ loop is the same, so a legal spot is
    found at the first free draw or given up after ``max_tries`` overlapping ones.
    """
    tries = 0
    while tries < max_tries:
        xn, yn = _random_point(rng, x0, y0, radius, float(r[c]))
        if not _overlaps(c, xn, yn, x, y, r):
            x[c], y[c] = xn, yn
            return True
        tries += 1
    return False


def distribute_components(
    x0: float,
    y0: float,
    radius: float,
    weights: np.ndarray,
    alpha: float,
    beta: float,
    log_mode: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack ``len(weights)`` discs inside the disc of radius ``radius`` at ``(x0, y0)``.

    Port of ``distribute_components``: every disc starts with radius
    ``R sqrt(alpha w^beta)`` (``w`` the normalized weight) at a random spot; discs
    outside the container or overlapping another are moved to a random free spot (up to
    ``10 N`` tries each, the smaller of an overlapping pair first); while every disc could
    be settled, ``alpha`` grows by 1 % and the discs are inflated in place (pulled
    towards the center by the radius increase); when a disc cannot be settled the last
    round is undone by dividing ``alpha`` by 1.1.

    The C++ re-seeded its generator with ``-seed`` on every call, so every packing of the
    same weights is the same; that is reproduced with a fresh generator per call.

    Deliberate deviation: the C++ pulled the inflated discs towards the origin of the
    picture, not the container center (harmless for the root, wrong for nested
    components, which the next round pushed back at random); here they are pulled
    towards ``(x0, y0)``.

    Parameters
    ----------
    x0, y0 : float
        Center of the container
    radius : float
        Radius of the container
    weights : np.ndarray
        One positive weight per disc (normalized here)
    alpha, beta : float
        Constant and exponent of the disc area law
    log_mode : bool
        ``log`` coordinate distribution (``log(1 + w)`` instead of ``w``)
    seed : int
        Seed of the random generator

    Returns
    -------
    (x, y, r) : tuple of np.ndarray
        Centers and radii of the discs
    """
    n = len(weights)
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    max_tries = 10 * n

    r = packing_radii(radius, w, alpha, beta, log_mode)
    if n < 2:
        # A lone disc never fails to settle, so the C++ loop would inflate it forever;
        # the caller gives a single child the whole container (findCoordinatesModern)
        return np.full(n, x0), np.full(n, y0), r
    x = np.empty(n)
    y = np.empty(n)
    for i in range(n):
        x[i], y[i] = _random_point(rng, x0, y0, radius, float(r[i]))

    while True:
        finish = False
        # Discs out of the container go to a random new position
        for i in range(n):
            if math.hypot(x[i] - x0, y[i] - y0) > radius - r[i] and not _give_new_random_position(
                i, x, y, r, x0, y0, radius, max_tries, rng
            ):
                finish = True
                break
        # Overlapping pairs: the smaller disc moves, then the larger if it could not
        for i in range(n - 1):
            for j in range(i + 1, n):
                if math.hypot(x[i] - x[j], y[i] - y[j]) < r[i] + r[j]:
                    c1, c2 = (i, j) if r[i] < r[j] else (j, i)
                    if not _give_new_random_position(
                        c1, x, y, r, x0, y0, radius, max_tries, rng
                    ) and not _give_new_random_position(
                        c2, x, y, r, x0, y0, radius, max_tries, rng
                    ):
                        finish = True
                        break
            if finish:
                break
        if finish:
            break
        # Everything fits: inflate the discs and pull them in by the increase
        alpha *= INCREMENT
        grown = packing_radii(radius, w, alpha, beta, log_mode)
        delta_r = grown - r
        r = grown
        dist = np.hypot(x - x0, y - y0)
        angle = np.arctan2(y - y0, x - x0)
        x = x0 + (dist - delta_r) * np.cos(angle)
        y = y0 + (dist - delta_r) * np.sin(angle)

    alpha /= FACTOR_CORRECTOR
    r = packing_radii(radius, w, alpha, beta, log_mode)
    return x, y, r
