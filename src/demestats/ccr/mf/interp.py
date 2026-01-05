import diffrax as dfx
import jax.numpy as jnp
from jaxtyping import ScalarLike

import demestats.util as util

from ...iicr.interp import Interpolator


class MeanFieldInterp(Interpolator):
    sol: dfx.Solution

    def jumps(self, demo):
        return jnp.array([])

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        t = jnp.clip(t, self.t0, self.t1)
        (r, b, log_s_seg) = self.sol.evaluate(t)
        etas = util.coalescent_rates(demo)
        eta = jnp.array([1.0 / (2.0 * etas[pop](t)) for pop in self.state.pops])
        c = jnp.sum(eta * r * b)
        return dict(c=c, log_s=self.state.log_s + log_s_seg, r=r, b=b)


class MeanFieldTerminalInterp(Interpolator):
    # Terminal segment with t1=inf: solve on-demand for each query time.

    def jumps(self, demo):
        return jnp.array([])

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        # On-demand solve from t0 to t.
        from .lift import _solve_segment

        t = jnp.clip(t, self.t0, t)
        sol = _solve_segment(self.state, self.t0, t, demo)
        (r, b, log_s_seg) = sol.evaluate(t)
        etas = util.coalescent_rates(demo)
        eta = jnp.array([1.0 / (2.0 * etas[pop](t)) for pop in self.state.pops])
        c = jnp.sum(eta * r * b)
        return dict(c=c, log_s=self.state.log_s + log_s_seg, r=r, b=b)
