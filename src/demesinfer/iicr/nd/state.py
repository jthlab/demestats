from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Int, Scalar, ScalarLike
from penzai import pz

from demesinfer import util

from ..state import SetupState, State


class StateNd(State):
    # p is an (n+1,)*d  array denoting the joint probability that each of N lineages is
    # in each of D demes.
    p: pz.nx.NamedArray

    @property
    def pops(self) -> tuple[str, ...]:
        return tuple(self.p.named_axes.keys())

    @property
    def n(self) -> int:
        ret = next(iter(self.p.named_axes.values()))
        assert all(v == ret for v in self.p.named_axes.values()), (
            f"Expected all axes to have the same size, got {self.p.named_axes}"
        )
        return ret - 1

    @property
    def d(self) -> int:
        return len(self.pops)

    @property
    def C(self) -> pz.nx.NamedArray:
        B = util.B_matrix(self.n, self.d)
        C = pz.nx.wrap(B * (B - 1) / 2, *self.pops, "n")
        return C

    def coal_rate(self, t, demo):
        etas = util.coalescent_rates(demo)
        eta = jnp.array([1 / 2 / etas[pop](t) for pop in self.pops])
        return pz.nx.nmap(jnp.dot)(self.C.untag("n"), eta)

    @classmethod
    def setup(
        cls, num_samples: dict[str, Int[ScalarLike, ""]], k: int
    ) -> dict[str, "StateNd"]:
        ret = {}
        for pop in num_samples:
            e = jnp.zeros((k + 1,))
            p = e.at[num_samples.get(pop, 0)].set(1.0)
            p = pz.nx.wrap(p, pop)
            ret[pop] = cls(p=p, log_s=jnp.array(0.0))
        return ret
