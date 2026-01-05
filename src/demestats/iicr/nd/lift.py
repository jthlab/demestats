from functools import partial

import diffrax as dfx
import jax.numpy as jnp
import numpy as np
from expm_unif import expm_multiply
from jax.experimental.sparse import BCOO, sparsify
from jaxtyping import Int, ScalarLike
from loguru import logger
from penzai import pz

import demestats.util as util
from demestats.bounded_solver import BoundedSolver
from demestats.numba.lift_nd_mats import mats

from .. import lift
from .interp import DfxInterp, ExpmNdInterp, PanmicticNdInterp
from .state import SetupState
from .state import StateNd as State


def setup(
    state: SetupState,
    t0i: Int[ScalarLike, ""],
    t1i: Int[ScalarLike, ""],
    terminal: bool,
    constant: bool,
    migrations: list[tuple[str, str]],
    aux: dict,
    demo: dict,
) -> tuple[SetupState, dict]:
    ret = {}
    ret["mats"] = mats(state.d, state.n)
    return state, ret


def _lift_expm(
    state: State, t0: ScalarLike, t1: ScalarLike, aux: dict, demo: dict
) -> tuple[State, dict]:
    logger.debug(
        "lifting {state} from {t0} to {t1} with expm", state=state, t0=t0, t1=t1
    )
    etas = util.coalescent_rates(demo)
    mats = aux["mats"]
    pops = state.pops
    mu = partial(util.migration_rate, demo)

    B = mats["B"]
    C = B * (B - 1) / 2

    t = (t0 + t1) / 2.0  # midpoint for rates
    eta = jnp.array([1 / 2 / etas[pop](t) for pop in pops])
    C_out = jnp.dot(C, eta)

    # migration matrix at time t
    M_t = jnp.array([[mu(dest, src, t) for dest in state.pops] for src in state.pops])
    # M_t = M_t - jnp.diag(M_t.sum(axis=1))  # make it a stochastic matrix

    # \sum_{j,u,v} p[j] N[i,j] B[j,u] M[u,v] X[i,j,u] Y[i,j,v]
    # find nonzero inds of U[i,j,u,v] = N[i,j] B[j,u] X[i,j,u] Y[i,j,v]

    # dp_in = \sum_{j,u,v} p[j] N[i,j] B[j,u] M[u,v] X[i,j,u] Y[i,j,v]
    # dp_out = \sum_{u,v}, p[i] B[i,u,v] M[u,v])
    # dp = dp_in - dp_out =
    #    = \sum_{j,u,v} p[j] N[i,j] B[j,u] M[u,v] X[i,j,u] Y[i,j,v] - \sum_{u,v}, p[i] B[i,u,v] M[u,v])
    #    = p'[i] = sum_j (p[j] U~[i,j] - p[i] B~[i])
    #    = \sum_j p[j] (U~[i,j] - delta_ij B~[i])
    # find nonzero inds of U[i,j,u,v] = N[i,j] B[j,u] X[i,j,u] Y[i,j,v]

    U = mats["U"]
    Q_in = (U * M_t * B[..., None]).sum((-2, -1))
    n = state.n
    d = state.d
    i = tuple(range(d))
    Q_out = jnp.einsum(B, i + (d + 1,), M_t, (d + 1, d + 2), i)
    # Now create a sparse matrix
    indices = []
    data = []
    for idx in np.ndindex(Q_out.shape):
        indices.append(idx * 2)  # i,j
        data.append(Q_out[idx] + C_out[idx])
    data = jnp.array(data, dtype=Q_in.data.dtype)
    indices = jnp.array(indices, dtype=Q_in.indices.dtype)
    Q_out = BCOO((data, indices), shape=Q_in.shape)
    Q = Q_in - Q_out
    # now reshape the sparse matrix to 2D
    assert Q.shape == (n + 1,) * (2 * state.d)
    Q_2d = Q.reshape(((n + 1) ** d, (n + 1) ** d))
    p_nd = state.p.unwrap(*pops)
    if (n + 1) ** d < 1000:
        logger.debug("switching to dense_expm for small migration matrix")
        Q_2d = Q_2d.todense()
    t = jnp.linspace(t0, t1, 100)
    res = expm_multiply(Q_2d, t - t0, p_nd.reshape(-1)).reshape(-1, *(p_nd.shape))
    p_prime = res[-1]
    p_nc = p_prime.sum()
    p_prime /= p_nc
    p1 = pz.nx.wrap(p_prime, *pops)
    f = ExpmNdInterp(
        state=state,
        t0=t0,
        t1=t1,
        ts=pz.nx.wrap(t, "t"),
        ps=pz.nx.wrap(res, "t", *pops),
    )
    ## important: update state after creating interp; interp uses old state
    state = State(
        p=p1,
        log_s=state.log_s + jnp.log(p_nc),
    )
    return state, {"interp": f}


def _ode(t, y, args):
    p, s = y
    state, demo, mats = args
    pops = state.pops
    mu = partial(util.migration_rate, demo)
    etas = util.coalescent_rates(demo)

    B = mats["B"]
    C = B * (B - 1) / 2

    def rate(t):
        eta = jnp.array([1 / 2 / etas[pop](t) for pop in pops])
        r = jnp.dot(C, eta)
        return pz.nx.wrap(r, *pops)

    # migration matrix at time t
    M_t = jnp.array([[mu(dest, src, t) for dest in state.pops] for src in state.pops])
    # M_t = M_t - jnp.diag(M_t.sum(axis=1))  # make it a stochastic matrix

    ds = p * rate(t)
    pu = p.unwrap(*pops)

    @sparsify
    def f(U):
        d = state.d
        i = tuple(range(d))
        j = tuple(range(d, 2 * d))
        u = 2 * d + 1
        v = 2 * d + 2
        return jnp.einsum(
            pu,
            j,
            U,
            i + j + (u, v),
            # N, i + j,
            B,
            j + (u,),
            # X, i + j + (u,),
            # Y, i + j + (v,),
            M_t,
            (u, v),
            i,
        )

    dp_in = f(mats["U"])
    dp_in = pz.nx.wrap(dp_in, *pops)
    #   B[j1,...,jd,u] = ju for u=1,...,d
    d = state.d
    i = tuple(range(d))
    dp_out = jnp.einsum(pu, i, B, i + (d + 1,), M_t, (d + 1, d + 2), i)
    dp_out = pz.nx.wrap(dp_out, *pops)
    dp = dp_in - dp_out
    # movement into coalescent state
    dp -= ds  # movement among migrant states, independent
    return dp, ds.unwrap(*pops).sum()


def _lift_ode(
    state: State, t0: ScalarLike, t1: ScalarLike, aux: dict, demo: dict
) -> tuple[State, dict]:
    """Lift partial likelihood.

    Args:
        st: state just before the lifting event
        demo: dict of parameters
        aux: dict of auxiliary data generated by setup

    Returns:
        State after the lifting event.
    """
    logger.debug("lifting {state} with ode", state=state)
    mats = aux["mats"]

    C = mats["B"] * (mats["B"] - 1) / 2
    C = pz.nx.wrap(C, *state.pops, "n")

    args = (state, demo, mats)
    etas = util.coalescent_rates(demo)
    eta_ts = jnp.concatenate([eta.t[:-1] for eta in etas.values()])
    mu_ts = jnp.array(
        [m.get(x, t0) for m in demo["migrations"] for x in ("start_time", "end_time")]
    )
    jump_ts = jnp.concatenate([eta_ts, mu_ts])
    jump_ts = jnp.sort(jump_ts)
    saveat = dfx.SaveAt(dense=True, t1=True)

    ssc = dfx.PIDController(rtol=1e-8, atol=1e-8, jump_ts=jump_ts)
    y0 = (state.p, 0.0)

    # if f has error, this will throw more comprehensibly than doing it inside of diffeqsolve
    _ = _ode(t0, y0, args)

    term = dfx.ODETerm(_ode)

    def oob_fn(y):
        p, s = y
        p = p.unwrap(*state.pops)
        eps = 1e-8
        return jnp.any(p < -eps) | jnp.any(p > 1 + eps) | (s < -eps) | (s > 1 + eps)

    solver = BoundedSolver(oob_fn=oob_fn)
    sol = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) / 1000.0,
        args=args,
        y0=y0,
        stepsize_controller=ssc,
        saveat=saveat,
        max_steps=4096,
    )
    p1 = sol.ys[0][0]
    s1 = sol.ys[1][0]
    p1 = (
        p1 / p1.unwrap(*state.pops).sum()
    )  # normalize to probability conditional on non-coalescence

    # interp interpolates from the "bottom" state up to t \in [t0, t1]
    interp = DfxInterp(sol=sol, state=state, t0=t0, t1=t1, jump_ts=jump_ts)

    # state update come *after* interp is created
    s1 = jnp.clip(s1, a_min=0.0, a_max=1.0 - 1e-7)  # avoid log(0)
    state = State(
        p=p1,
        log_s=state.log_s + jnp.log1p(-s1),
    )

    return state, {"interp": interp}


execute = partial(
    lift.execute,
    PanmicticInterp=PanmicticNdInterp,
    lift_ode=_lift_ode,
    lift_expm=_lift_expm,
)
