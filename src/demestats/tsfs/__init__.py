from dataclasses import dataclass, field

import demes
import jax
import jax.numpy as jnp
import numpy as np
from beartype.typing import Sequence
from jaxtyping import Array, Float, Int, ScalarLike

import demestats.event_tree as event_tree
from demestats.sfs import ExpectedSFS, Params, PruneInput, _call


@dataclass
class ExpectedTSFS:
    """
    Build an ExpectedTSFS object that can be used to compute a time-stratified site
    frequency spectrum on a fixed time grid.

    Parameters
    ----------
        demo : demes.Graph
            A ``demes`` graph.
        num_samples : dict
            Number of haploid samples per population.
        time_grid : sequence
            A strictly increasing time discretization with
            ``time_grid[0] = 0`` and ``time_grid[-1] = +inf``.
        prune : mapping or sequence, optional
            Optional pruning specification, as in ``demestats.sfs.ExpectedSFS``.
    """

    demo: demes.Graph
    num_samples: dict[str, Int[ScalarLike, ""]]
    time_grid: Sequence[ScalarLike]
    prune: PruneInput = None
    _sfs: ExpectedSFS = field(init=False, repr=False)
    _time_grid: Float[Array, "M+1"] = field(init=False, repr=False)

    def __post_init__(self):
        self._time_grid = _validate_time_grid(self.time_grid)
        self._sfs = ExpectedSFS(self.demo, self.num_samples, prune=self.prune)

    @property
    def et(self) -> event_tree.EventTree:
        return self._sfs.et

    def bind(self, params: Params) -> dict:
        demo = self._sfs.bind(params)
        demo["_time_grid"] = self._time_grid / self.et.scaling_factor
        return demo

    def variable_for(self, path):
        return self._sfs.variable_for(path)

    @property
    def variables(self):
        return self._sfs.variables

    def __call__(self, params: Params = {}) -> Float[Array, "M *T"]:
        bs = [n + 1 for n in self.num_samples.values()]
        num_derived = jnp.indices(bs)
        num_derived = jnp.rollaxis(num_derived, 0, num_derived.ndim).reshape(
            -1, len(bs)
        )

        def f(nd):
            nd = dict(zip(self.num_samples, nd))
            ret = {}
            for pop, leaf in self.et.leaves.items():
                del leaf
                n = self.num_samples.get(pop, 0)
                d = nd.get(pop, 0)
                ret[pop] = jax.nn.one_hot(jnp.array([d]), n + 1)[0]
            return ret

        X = jax.vmap(f)(num_derived)
        res = self.dp(params, X)
        res = res.at[jnp.array([0, -1])].set(0.0)
        return jnp.moveaxis(res.reshape(*bs, res.shape[-1]), -1, 0)

    def dp(
        self,
        params: Params,
        X: dict[str, Float[Array, "batch *T"]],
    ) -> Float[Array, "batch M"]:
        demo = self.bind(params)
        state = _call(
            X,
            self.et,
            demo,
            self._sfs._aux,
            jnp.zeros(self._time_grid.shape[0] - 1),
        )
        return state.phi * self.et.scaling_factor


def _validate_time_grid(time_grid: Sequence[ScalarLike]) -> Array:
    grid = np.asarray(time_grid, dtype=float)
    if grid.ndim != 1:
        raise ValueError("time_grid must be a one-dimensional array")
    if grid.shape[0] < 2:
        raise ValueError("time_grid must contain at least two points")
    if not np.isclose(grid[0], 0.0):
        raise ValueError("time_grid must start at 0")
    if not np.isinf(grid[-1]):
        raise ValueError("time_grid must end at +inf")
    if not np.all(np.isfinite(grid[:-1])):
        raise ValueError("all time_grid entries except the last must be finite")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing")
    return jnp.asarray(grid)


__all__ = ["ExpectedTSFS"]
