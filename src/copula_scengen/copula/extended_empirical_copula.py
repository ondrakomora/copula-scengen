from functools import cached_property
from itertools import combinations

import numpy as np

from copula_scengen.copula.base import Copula
from copula_scengen.copula.empirical_copula import EmpiricalCopula
from copula_scengen.functions.is_discrete import is_discrete

_TOLERANCE = 1e-9


class ExtendedEmpiricalCopula(Copula):
    """
    Empirical copula for a mixed random vector with continuous and discrete margins.

    Implements the discrete-extension formula: given any copula ``C`` satisfying Sklar's
    theorem for the mixed vector, the copula ``C*`` of the (continuized) discrete extension is

        C*(u, v) = sum_{S subset of discrete margins}
            C(u, v^S) * prod_{i in S} lambda_i(v_i) * prod_{j not in S} (1 - lambda_j(v_j))

    where v_i^S = v_i^+ if i in S else v_i^-, with v_i^-, v_i^+ the lower/upper "steps"
    (nearest attained CDF values) around v_i, and lambda_i the linear interpolation weight
    between them (zero when v_i is itself an attained CDF value).
    """

    def __init__(self, data: np.ndarray) -> None:
        self.data = data
        self._inner_copula = EmpiricalCopula(data=self.data)
        self._discrete_margins = [j for j in range(self.data.shape[1]) if is_discrete(self.data[:, j])]

    @cached_property
    def _jump_points(self) -> dict[int, np.ndarray]:
        """For each discrete margin, the sorted, distinct, attained CDF values (range of F), with a leading 0.0."""
        jump_points = {}
        n = self.data.shape[0]
        for j in self._discrete_margins:
            column = self.data[:, j]
            min_val = int(np.min(column))
            value_counts = np.bincount((column - min_val).astype(int))
            cumulative = np.cumsum(value_counts) / n
            jump_points[j] = np.concatenate(([0.0], cumulative))
        return jump_points

    def _steps(self, j: int, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute v^- and v^+ (lower/upper steps) for margin j, vectorized over query points v."""
        points = self._jump_points[j]

        # v^- = sup{p in points : p <= v}; v^+ = inf{p in points : p >= v}
        lower_idx = np.searchsorted(points, v, side="right") - 1
        lower_idx = np.clip(lower_idx, 0, points.size - 1)

        upper_idx = np.searchsorted(points, v, side="left")
        # if v exactly matches a jump point, searchsorted(side="left") already lands on it
        upper_idx = np.clip(upper_idx, 0, points.size - 1)

        return points[lower_idx], points[upper_idx]

    def _lambda(self, j: int, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (v^-, v^+, lambda_j(v)) for margin j, guarding against division by zero."""
        lower, upper = self._steps(j, v)
        span = upper - lower
        weight = np.divide(v - lower, span, out=np.zeros_like(v), where=span > _TOLERANCE)
        return lower, upper, np.clip(weight, 0.0, 1.0)

    def _subset_term(
        self, args: np.ndarray, lower: np.ndarray, upper: np.ndarray, weight: np.ndarray, subset: tuple[int, ...]
    ) -> np.ndarray:
        query = args.copy()
        subset_weight = np.ones(args.shape[0], dtype=float)
        for j in self._discrete_margins:
            if j in subset:
                query[:, j] = upper[:, j]
                subset_weight *= weight[:, j]
            else:
                query[:, j] = lower[:, j]
                subset_weight *= 1.0 - weight[:, j]
        return subset_weight * self._inner_copula(query)

    def __call__(self, args: np.ndarray) -> np.ndarray:
        if args.ndim == 1:
            args = args[None, :]

        if not self._discrete_margins:
            return self._inner_copula(args)

        lower = np.empty_like(args)
        upper = np.empty_like(args)
        weight = np.empty_like(args)
        for j in self._discrete_margins:
            lower[:, j], upper[:, j], weight[:, j] = self._lambda(j, args[:, j])

        result = np.zeros(args.shape[0], dtype=float)
        for subset_size in range(len(self._discrete_margins) + 1):
            for subset in combinations(self._discrete_margins, subset_size):
                result += self._subset_term(args=args, lower=lower, upper=upper, weight=weight, subset=subset)

        return result

    def grid(self, max_rank: int) -> np.ndarray:
        """
        Evaluate the copula on the ``(i / max_rank, r / max_rank)`` lattice, exactly and quickly.

        Restricted to the 2-margin case used by the generator. The inner empirical copula is
        evaluated a single time on the (small) product of per-axis query values -- lattice
        coordinates for continuous axes, jump points for discrete axes -- via its cumulative
        histogram, instead of a full broadcast per subset. Falls back to the batched
        ``__call__`` for other dimensionalities.
        """
        coords = np.arange(max_rank + 1) / max_rank

        if self.data.shape[1] != 2:  # noqa: PLR2004
            first, second = np.meshgrid(coords, coords, indexing="ij")
            args = np.column_stack((first.ravel(), second.ravel()))
            return self(args).reshape(max_rank + 1, max_rank + 1)

        if not self._discrete_margins:
            return self._inner_copula.grid(max_rank)

        # Per axis: the sorted query values fed to the inner copula, plus, for each lattice
        # coordinate, the index of its lower/upper step within those values and the weight.
        axis_values: list[np.ndarray] = []
        lower_idx: list[np.ndarray] = []
        upper_idx: list[np.ndarray] = []
        weights: list[np.ndarray] = []
        for axis in range(2):
            if axis in self._discrete_margins:
                low, up, wgt = self._lambda(axis, coords)
                values = self._jump_points[axis]
                axis_values.append(values)
                lower_idx.append(np.searchsorted(values, low))
                upper_idx.append(np.searchsorted(values, up))
                weights.append(wgt)
            else:
                axis_values.append(coords)
                identity = np.arange(coords.size)
                lower_idx.append(identity)
                upper_idx.append(identity)
                weights.append(None)

        inner_grid = self._inner_copula.cumulative_counts(axis_values)

        result = np.zeros((max_rank + 1, max_rank + 1), dtype=float)
        for subset_size in range(len(self._discrete_margins) + 1):
            for subset in combinations(self._discrete_margins, subset_size):
                idx0 = upper_idx[0] if 0 in subset else lower_idx[0]
                idx1 = upper_idx[1] if 1 in subset else lower_idx[1]

                factor0 = (
                    np.ones(max_rank + 1)
                    if 0 not in self._discrete_margins
                    else (weights[0] if 0 in subset else 1.0 - weights[0])
                )
                factor1 = (
                    np.ones(max_rank + 1)
                    if 1 not in self._discrete_margins
                    else (weights[1] if 1 in subset else 1.0 - weights[1])
                )

                result += np.outer(factor0, factor1) * inner_grid[np.ix_(idx0, idx1)]

        return result
