"Likelihood of an ARG"

import diffrax as dfx
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import xlog1py, xlogy
from jaxtyping import Array, Float, Scalar, ScalarLike

from .coal_rate import CoalRate


def loglik(eta: CoalRate, r: ScalarLike, data: Float[Array, "intervals 2"]) -> Scalar:
    """Compute the log-likelihood of the data given the demographic model.

    Args:
        eta: Coalescent rate at time t.
        r: float, the recombination rate.
        data: the data to compute the likelihood for. The first column is the TMRCA, and the second column is the span.

    Notes:
        - Successive spans that have the same TMRCA should be merged into one span:
          <tmrca, span1> + <tmrca, span1> = <tmrca, span + span>.
        - Missing data/padding indicated by span<=0.
    """

    def p0(t0, span, t1):
        return 0.0

    def p1(t0, span, t1):
        def f(t, y, _):
            c = eta(t)
            A1 = jnp.array([[-r, r, 0.0], [c, -2 * c, c], [0.0, 0.0, 0.0]])
            A2 = jnp.array([[0.0, 0.0, 0.0], [0.0, -c, c], [0.0, 0.0, 0.0]])
            A = jnp.where(t < t0, A1, A2)
            return A.T @ y

        y0 = jnp.array([1.0, 0.0, 0.0])
        solver = dfx.Tsit5()
        term = dfx.ODETerm(f)
        jumps = jnp.append(eta.jumps, t0)
        ssc = dfx.PIDController(rtol=1e-6, atol=1e-6, jump_ts=jumps)
        if t1 is None:
            # we only know that the span extended at least this far
            sol = dfx.diffeqsolve(
                term, solver, 0.0, t0, dt0=t0 / 100.0, y0=y0, stepsize_controller=ssc
            )
            return xlogy(span, sol.ys[0, 0])

        m = jnp.minimum(t0, t1)
        M = jnp.maximum(t0, t1)

        check = f(0.0, y0, None)

        sol = dfx.diffeqsolve(
            term,
            solver,
            0.0,
            M,
            dt0=t0 / 100.0,
            y0=y0,
            stepsize_controller=ssc,
            saveat=dfx.SaveAt(t1=True, ts=[m]),
        )
        (p_nr_m, p_float_m, p_coal_m), (p_nr_M, p_float_M, p_coal_M) = sol.ys
        return jnp.where(
            t0 < t1,
            xlog1py(span - 1, -p_nr_m) + jnp.log(p_coal_M),
            xlog1py(span - 1, -p_nr_M) + jnp.log(p_coal_m),
        )

    def p(t0, span, t1):
        return jax.lax.cond(span <= 0, p0, p1, t0, span, t1)

    times, spans = data.T
    t0 = times[:-1]
    t1 = times[1:]
    ret = jax.vmap(p)(t0, spans[:-1], t1).sum()
    ret += p(times[-1], spans[-1], None)
    return ret
