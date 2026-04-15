from functools import partial

import diffrax as dfx
import jax.numpy as jnp
from jaxtyping import Int, ScalarLike

import demestats.util as util

from .interp import MeanFieldInterp, MeanFieldTerminalInterp
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


def _rhs(t, y, args):
    u, log_s_seg = y
    state, demo = args
    pops = state.pops
    mu = partial(util.migration_rate, demo)

    M = jnp.array([[mu(dest, src, t) for dest in pops] for src in pops], dtype=u.dtype)
    M = M - jnp.diag(M.sum(axis=1))
    du = u @ M
    c = _hazard(t, state=state, u=u, demo=demo)
    return (du, -c)


def _solve_segment(
    state: State, t0: ScalarLike, t1: ScalarLike, demo: dict
) -> dfx.Solution:
    term = dfx.ODETerm(_rhs)
    y0 = (state.u, jnp.array(0.0, dtype=state.u.dtype))
    solver = dfx.Tsit5()
    saveat = dfx.SaveAt(dense=True, t1=True)
    stepsize_controller = dfx.PIDController(rtol=1e-6, atol=1e-6)
    return dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) / 16.0,
        y0=y0,
        args=(state, demo),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=4096,
    )


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
    del constant, migrations, aux
    t0 = demo["_times"][t0i]
    t1 = demo["_times"][t1i]

    if jnp.isinf(t1):
        interp = MeanFieldTerminalInterp(state=state, t0=t0, t1=t1)
        return state, {"interp": interp}

    sol = _solve_segment(state, t0, t1, demo)
    u1, log_s_seg1 = sol.evaluate(t1)
    interp = MeanFieldInterp(state=state, t0=t0, t1=t1, sol=sol)
    if terminal:
        return state, {"interp": interp}
    state1 = State(
        pops=state.pops,
        group_sizes=state.group_sizes,
        u=u1,
        log_s=state.log_s + log_s_seg1,
    )
    return state1, {"interp": interp}
