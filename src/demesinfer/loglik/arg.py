"Likelihood of an ARG"

import diffrax as dfx
import jax
import jax.numpy as jnp
from beartype.typing import Callable
from jax import vmap
from jax.scipy.special import xlog1py, xlogy
from jaxtyping import Array, Float, Scalar, ScalarLike

from ..bounded_solver import BoundedSolver


def loglik(
    eta: Callable[[ScalarLike], ScalarLike],
    r: ScalarLike,
    data: Float[Array, "intervals 2"],
    max_index: Array,
    jump_ts: Float[Array, "T"] = None,
) -> Scalar:
    """Compute the log-likelihood of the data given the demographic model.

    Args:
        eta: Coalescent rate at time t.
        r: float, the recombination rate.
        data: the data to compute the likelihood for. The first column is the TMRCA, and
              the second column is the span.

    Notes:
        - Successive spans that have the same TMRCA should be merged into one span:
          <tmrca, span1> + <tmrca, span1> = <tmrca, span + span>.
        - Missing data/padding indicated by span<=0.
    """
    times, spans = data.T
    i = times.argsort()
    sorted_times = times[i]

    def f(t, y, _):
        c = eta(t)
        A = jnp.array([[-r, r, 0.0], [c, -2 * c, c], [0.0, 0.0, 0.0]])
        return A.T @ y

    y0 = jnp.array([1.0, 0.0, 0.0])

    def oob_fn(y):
        return jnp.any(y < 0.0) | jnp.any(y > 1.0)

    solver = BoundedSolver(oob_fn=oob_fn)
    term = dfx.ODETerm(f)
    ssc = dfx.PIDController(rtol=1e-6, atol=1e-6, jump_ts=jump_ts)
    T = times.max()
    sol = dfx.diffeqsolve(
        term,
        solver,
        0.0,
        T,
        dt0=0.001,
        y0=y0,
        stepsize_controller=ssc,
        saveat=dfx.SaveAt(ts=sorted_times),
    )

    # invert the sorting so that cscs matches times
    i_inv = i.argsort()
    cscs = sol.ys[i_inv]

    @vmap
    def p(t0, csc0, t1, csc1, span):
        p_nr_t0, p_float_t0, p_coal_t0 = csc0
        p_nr_t1, p_float_t1, p_coal_t1 = csc1
        # no recomb for first span - 1 positions
        r1 = xlogy(span - 1, p_nr_t0)
        # coalescence at t1
        r2 = jnp.log(eta(t1))
        # back-coalescence process up to t1, depends to t0 >< t1
        r3 = jnp.where(
            t0 < t1, jnp.log(p_float_t0) - eta.R(t0, t1), jnp.log(p_float_t1)
        )
        return r1 + r2 + r3

    ll = p(times[:-1], cscs[:-1], times[1:], cscs[1:], spans[:-1])
    ll = jnp.dot(ll, jnp.arange(len(times[:-1])) < max_index)

    # for the last position, we only know span was at least as long
    ll += xlogy(spans[max_index], cscs[max_index, 0])
    return ll
