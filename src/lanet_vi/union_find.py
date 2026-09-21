"""Disjoint sets, the workhorse of the one-pass component constructions."""

from __future__ import annotations


class UnionFind:
    """Disjoint sets over ``0 .. n - 1`` with path halving and union by size."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, i: int) -> int:
        """Return the representative of the set holding ``i``."""
        parent = self.parent
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(self, i: int, j: int) -> None:
        """Merge the sets holding ``i`` and ``j``."""
        i, j = self.find(i), self.find(j)
        if i == j:
            return
        if self.size[i] < self.size[j]:
            i, j = j, i
        self.parent[j] = i
        self.size[i] += self.size[j]
