from functools import partial

import diffrax as dfx
import jax
import jax.numpy as jnp
from jaxtyping import Int, ScalarLike

import demestats.util as util

from .interp import (
    MeanFieldInterp,
)
from .state import StateMf as State


def _rhs(t, y, args):
    r, b, log_s = y
    state, demo = args
    pops = state.pops
    mu = partial(util.migration_rate, demo)
    etas = util.coalescent_rates(demo)

    M = jnp.array([[mu(dest, src, t) for dest in pops] for src in pops])
    M = M - jnp.diag(M.sum(axis=1))

    # migration
    dr = r @ M
    db = b @ M

    # same-color coalescence
    eta = jnp.array([1.0 / (2.0 * etas[pop](t)) for pop in pops])
    dr = dr - eta * (r * (r - 1.0) / 2.0)
    db = db - eta * (b * (b - 1.0) / 2.0)

    # first cross event hazard
    rate_cross = eta * r * b
    dlog_s = -jnp.sum(rate_cross)

    # repulsion from conditioning on no cross event
    dr = dr - rate_cross
    db = db - rate_cross
    return (dr, db, dlog_s)


def _solve_segment(
    state: State, t0: ScalarLike, t1: ScalarLike, demo: dict
) -> dfx.Solution:
    term = dfx.ODETerm(_rhs)
    y0 = (state.r, state.b, jnp.array(0.0, dtype=state.r.dtype))
    # Default solver: nonstiff RK; segments are typically short due to event tree splitting.
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(dense=True, t1=True)
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-6)

    # If t0 == t1, dt0 would be 0.0, which can cause issues.
    # diffeqsolve handles t0=t1 by returning y0 at t1, but dt0=0.0 might be problematic.
    # Let's explicitly set dt0 to None to let the solver choose, or handle it.
    # For t0=t1, dt0=None is fine, as is (t1-t0)/16.0 if t1-t0 is not 0.
    # If t1-t0 is 0, dt0=0.0. Diffrax handles this by returning y0 at t1.
    # So, the original dt0 calculation is fine.
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) / 16.0,
        y0=y0,
        args=(state, demo),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        # NOTE: Diffrax allocates internal buffers sized by `max_steps`. Using a huge
        # value can immediately exhaust GPU memory even for tiny state dimensions.
        max_steps=4096,
    )
    return sol


def _exec_inf(state, t0, t1, demo):
    # Dummy solution for structure consistency (solve t0->t0)
    sol = _solve_segment(state, t0, t0, demo)
    interp = MeanFieldInterp(state=state, t0=t0, t1=t1, sol=sol)

    # Cast state components to match state.r.dtype to ensure consistency
    dtype = state.r.dtype
    log_s = state.log_s.astype(dtype)
    r = state.r.astype(dtype)
    b = state.b.astype(dtype)
    state_out = State(pops=state.pops, r=r, b=b, log_s=log_s)

    return state_out, {"interp": interp}


def _exec_finite(state, t0, t1, demo, terminal):
    sol = _solve_segment(state, t0, t1, demo)
    r1, b1, log_s_seg1 = sol.evaluate(t1)

    dtype = state.r.dtype
    r1 = r1.astype(dtype)
    b1 = b1.astype(dtype)
    log_s_seg1 = log_s_seg1.astype(dtype)

    interp = MeanFieldInterp(state=state, t0=t0, t1=t1, sol=sol)

    if terminal:
        # Cast state components to match state.r.dtype
        log_s = state.log_s.astype(dtype)
        r = state.r.astype(dtype)
        b = state.b.astype(dtype)
        state_out = State(pops=state.pops, r=r, b=b, log_s=log_s)
        return state_out, {"interp": interp}

    # log_s_seg1 is already cast, but ensure sum doesn't promote
    log_s_next = (state.log_s + log_s_seg1).astype(dtype)

    state1 = State(pops=state.pops, r=r1, b=b1, log_s=log_s_next)
    return state1, {"interp": interp}


def execute(
    state: State,
    t0i: Int[ScalarLike, ""],
    t1i: Int[ScalarLike, ""],
    terminal: bool,
    constant: bool,
    migrations: list[tuple[str, str]],
    aux: dict,
    demo: dict,
) -> tuple[State, dict]:
    t0 = demo["_times"][t0i]
    t1 = demo["_times"][t1i]

    return jax.lax.cond(
        jnp.isinf(t1),
        partial(_exec_inf, state=state, t0=t0, t1=t1, demo=demo),
        partial(_exec_finite, state=state, t0=t0, t1=t1, demo=demo, terminal=terminal),
    )
