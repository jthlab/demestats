import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import ScalarLike

import demestats.util as util

from ...iicr.interp import Interpolator


class MeanFieldInterp(Interpolator):
    sol: dfx.Solution

    def jumps(self, demo):
        return jnp.array([])

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        # If t1 is infinite, we solve on demand from t0 to t.
        # Otherwise, we interpolate using the pre-computed solution.
        def _eval_sol(t, _):
            t = jnp.clip(t, self.t0, self.t1)
            return self.sol.evaluate(t)

        def _eval_ondemand(t, _):
            from .lift import _solve_segment

            # For on-demand, t1 in the solve segment is just t.
            # We must clip t >= t0.
            t_eval = jnp.maximum(t, self.t0)
            # Use t_eval + epsilon? No, t_eval is fine.
            # We solve from t0 to t.
            sol = _solve_segment(self.state, self.t0, t_eval, demo)
            return sol.evaluate(t_eval)

        # Use lax.cond to switch behavior based on t1
        (r, b, log_s_seg) = jax.lax.cond(
            jnp.isinf(self.t1), _eval_ondemand, _eval_sol, t, None
        )
        etas = util.coalescent_rates(demo)
        eta = jnp.array([1.0 / (2.0 * etas[pop](t)) for pop in self.state.pops])
        c = jnp.sum(eta * r * b)
        return dict(c=c, log_s=self.state.log_s + log_s_seg, r=r, b=b)
