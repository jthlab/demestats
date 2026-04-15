import diffrax as dfx
import jax.numpy as jnp
from jaxtyping import ScalarLike

import demestats.util as util

from ..interp import Interpolator
from .state import StateMf as State


def _hazard(t: ScalarLike, *, state: State, u, demo: dict) -> ScalarLike:
    etas = util.coalescent_rates(demo)
    eta = jnp.array([1.0 / (2.0 * etas[pop](t)) for pop in state.pops], dtype=u.dtype)

    u2 = u * u
    n = state.group_sizes.astype(u.dtype)
    w = jnp.where(n > 0, (n - 1.0) / (2.0 * n), 0.0)
    within = (w[:, None] * u2).sum(axis=0)

    u_sum = u.sum(axis=0)
    u_sq_sum = u2.sum(axis=0)
    cross = (u_sum * u_sum - u_sq_sum) / 2.0
    pairs = cross + within
    return jnp.sum(eta * pairs)


class MeanFieldInterp(Interpolator):
    sol: dfx.Solution

    def jumps(self, demo):
        return jnp.array([])

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        t = jnp.clip(t, self.t0, self.t1)
        u, log_s_seg = self.sol.evaluate(t)
        c = _hazard(t, state=self.state, u=u, demo=demo)
        return dict(c=c, log_s=self.state.log_s + log_s_seg, u=u)


class MeanFieldTerminalInterp(Interpolator):
    # Terminal segment with t1=inf: solve on-demand for each query time.

    def jumps(self, demo):
        return jnp.array([])

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        from .lift import _solve_segment

        t = jnp.clip(t, self.t0, t)
        sol = _solve_segment(self.state, self.t0, t, demo)
        u, log_s_seg = sol.evaluate(t)
        c = _hazard(t, state=self.state, u=u, demo=demo)
        return dict(c=c, log_s=self.state.log_s + log_s_seg, u=u)
