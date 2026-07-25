from functools import cached_property

import numpy as np

from copula_scengen.copula.base import Copula
from copula_scengen.functions.pseudoobservations import compute_pseudoobservations


class EmpiricalCopula(Copula):
    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    @cached_property
    def pseudo_observations(self) -> np.ndarray:
        per_margin = [compute_pseudoobservations(self.data[:, j]) for j in range(self.data.shape[1])]
        return np.vstack(per_margin).T.astype(float)

    def __call__(self, args: np.ndarray) -> np.ndarray:
        # allow (d,) -> (1, d)
        if args.ndim == 1:
            args = args[None, :]

        return np.mean((args[:, None, :] >= self.pseudo_observations[None, :, :]).all(axis=2), axis=1)

    def cumulative_counts(self, thresholds: list[np.ndarray]) -> np.ndarray:
        """
        Evaluate the empirical copula on the axis-aligned lattice spanned by ``thresholds``.

        ``thresholds[a]`` is a sorted 1D array of query values for axis ``a``. The returned
        array ``G`` has shape ``tuple(t.size for t in thresholds)`` with

            G[i_0, ..., i_{d-1}] == C((thresholds[0][i_0], ..., thresholds[d-1][i_{d-1}]))

        computed via a d-dimensional cumulative histogram of the pseudo-observations. This
        is ``O(n + prod(sizes))`` instead of the ``O(n * prod(sizes))`` broadcast in
        :meth:`__call__`, and is bit-identical because each point is binned at the smallest
        threshold index ``j`` with ``thresholds[a][j] >= pseudo`` (matching the ``>=`` test).
        """
        pseudo = self.pseudo_observations
        n, d = pseudo.shape
        shape = tuple(t.size for t in thresholds)

        # smallest index j on each axis with thresholds[a][j] >= pseudo; == size means the
        # point exceeds every threshold on that axis and therefore contributes nowhere.
        per_axis_idx = [np.searchsorted(thresholds[a], pseudo[:, a], side="left") for a in range(d)]
        valid = np.ones(n, dtype=bool)
        for a in range(d):
            valid &= per_axis_idx[a] < shape[a]

        counts = np.zeros(shape, dtype=np.int64)
        np.add.at(counts, tuple(idx[valid] for idx in per_axis_idx), 1)

        grid = counts
        for a in range(d):
            grid = np.cumsum(grid, axis=a)
        return grid / n

    def grid(self, max_rank: int) -> np.ndarray:
        """Evaluate the copula on the lattice ``(i / max_rank)`` per axis, exactly and quickly."""
        coords = np.arange(max_rank + 1) / max_rank
        return self.cumulative_counts([coords] * self.data.shape[1])
