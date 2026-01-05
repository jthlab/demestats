import os
from functools import partial

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from expm_unif import expm_multiply
from jaxtyping import Int, ScalarLike
from loguru import logger
from penzai import pz

import demestats.util as util
from demestats.numba.lift_nd_mats import mats_ccr

from .interp import ExpmCcrInterp
from .state import SetupState
from .state import StateCcr as State


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
    mats = mats_ccr(state.pops, state.n)
    ret["mats"] = mats
    ret["axes"] = mats["axes"]
    ret["pops"] = mats["pops"]
    ret["size"] = mats["size"]
    return state, ret


def _to_expm_coo(Q: sp.csr_matrix) -> tuple[jnp.ndarray, jnp.ndarray]:
    coo = Q.tocoo()
    indices = np.stack([coo.row, coo.col], axis=1).astype(np.int64)
    return jnp.asarray(coo.data), jnp.asarray(indices)


def _to_expm_repr(Q: sp.csr_matrix, *, size: int, dtype) -> object:
    # Use dense expm for small state spaces to match iicr.nd behavior and reduce
    # numerical differences between sparse/dense paths.
    if size < 1000:
        return jnp.asarray(Q.toarray(), dtype=dtype)
    data, indices = _to_expm_coo(Q)
    return (jnp.asarray(data, dtype=dtype), indices)


def _generator_matrix(aux: dict, demo: dict, t: float) -> sp.csr_matrix:
    pops = aux["pops"]
    size = aux["size"]
    Q = sp.csr_matrix((size, size))
    mu = partial(util.migration_rate, demo)
    etas = util.coalescent_rates(demo)
    eta = np.array([1 / 2 / etas[pop](t) for pop in pops], dtype=float)

    for color in ("red", "blue"):
        for (dest, src), block in mats["migration"][color].items():
            if block.nnz == 0:
                continue
            rate = mu(pops[dest], pops[src], t)
            rate = float(rate)
            if rate != 0.0:
                Q = Q + block * rate

    for deme_idx in range(len(pops)):
        coeff = eta[deme_idx]
        if coeff == 0.0:
            continue
        for color in ("red", "blue"):
            block = mats["coalesce"][color][deme_idx]
            if block.nnz:
                Q = Q + block * coeff
        cross_block = mats["cross"][deme_idx]
        if cross_block.nnz:
            Q = Q + cross_block * coeff
    return Q


def _lift_expm(
    state: State,
    t0: ScalarLike,
    t1: ScalarLike,
    aux: dict,
    demo: dict,
    *,
    t1_eval: ScalarLike | None = None,
    update_state: bool = True,
) -> tuple[State, dict]:
    logger.debug(
        "lifting {state} from {t0} to {t1} with expm", state=state, t0=t0, t1=t1
    )
    axes = aux["axes"]
    p0 = state.p.unwrap(*axes)
    vec0 = jnp.asarray(p0).reshape(-1)
    if t1_eval is None:
        t1_eval = t1
    t0f = float(t0)
    t1f = float(t1_eval)
    if not np.isfinite(t1f) or t1f <= t0f:
        raise ValueError(
            f"t1_eval must be finite and > t0; got t0={t0f}, t1_eval={t1f}"
        )

    # Time grid for interpolation: use log-spacing to resolve early-time behavior.
    grid_n = int(os.environ.get("DEMESTATS_CCR_EXPM_GRID", "128"))
    max_entries = int(os.environ.get("DEMESTATS_CCR_EXPM_MAX_ENTRIES", "20000000"))
    # Cap grid size to keep the temporary `res` array manageable: res has shape
    # (grid_n, nstates).
    if vec0.size:
        grid_cap = max(32, max_entries // int(vec0.size))
        grid_n = int(min(grid_n, grid_cap))
    dt_max = t1f - t0f
    if dt_max <= 0:
        dt = jnp.array([0.0], dtype=vec0.dtype)
    elif dt_max < 1.0:
        dt = jnp.linspace(0.0, dt_max, grid_n, dtype=vec0.dtype)
    else:
        dt_min = min(1e-3, dt_max * 1e-6)
        dt_pos = np.geomspace(dt_min, dt_max, grid_n - 1).astype(np.float32)
        dt = jnp.concatenate(
            [jnp.array([0.0], dtype=vec0.dtype), jnp.asarray(dt_pos, dtype=vec0.dtype)]
        )
    ts = t0 + dt

    t_mid = float(t0f + 0.5 * dt_max)
    Q = _generator_matrix(aux=aux, demo=demo, t=t_mid)
    Q = _to_expm_repr(Q, size=aux["size"], dtype=vec0.dtype)

    # Propagate unnormalised probabilities across the segment for a set of times.
    res = expm_multiply(Q, dt, vec0)  # shape (t, nstates)
    res = jnp.asarray(res)
    p_nc = res.sum(axis=1)
    log_pnc = jnp.where(p_nc > 0, jnp.log(p_nc), -jnp.inf)

    # Hazard curve on the grid: c(t) = E[rate(X_t) | no-cross].
    rate_vec = state.coal_rate(t_mid, demo).unwrap(*axes).reshape(-1)
    p_cond = jnp.where(p_nc[:, None] > 0, res / p_nc[:, None], 0.0)
    c_ts = (p_cond * rate_vec[None, :]).sum(axis=1)

    f = ExpmCcrInterp(state=state, t0=t0, t1=t1, ts=ts, cs=c_ts, log_pnc=log_pnc)

    if not update_state:
        return state, {"interp": f}

    # Update state at t1 using the endpoint of the grid (which is exactly t1 if finite).
    if float(t1) != float(t1_eval):
        # If we used a shortened evaluation horizon (terminal segment), do not update.
        return state, {"interp": f}

    p_nc_end = p_nc[-1]
    p1_unnorm = res[-1].reshape(p0.shape)
    p1_cond = jnp.where(p_nc_end > 0, p1_unnorm / p_nc_end, 0.0)
    p1 = pz.nx.wrap(p1_cond, *axes)
    state = State(p=p1, log_s=state.log_s + log_pnc[-1])
    return state, {"interp": f}


def _ode(t, y, args):
    raise NotImplementedError(
        "CCR ODE lifting is not implemented with the new generator."
    )


def _lift_ode(
    state: State, t0: ScalarLike, t1: ScalarLike, aux: dict, demo: dict
) -> tuple[State, dict]:
    raise NotImplementedError(
        "CCR ODE lifting is not implemented with the new generator."
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
    # Unlike the IICR, CCR has no simple closed-form panmictic interpolator because
    # red-red and blue-blue coalescences can occur before the first red-blue event.
    # For now we always use expm when the epoch is constant.
    t0 = demo["_times"][t0i]
    t1 = demo["_times"][t1i]
    if constant:
        if "axes" not in aux:
            mats = mats_ccr(state.pops, state.n)
            aux = dict(
                axes=mats["axes"], pops=mats["pops"], mats=mats, size=mats["size"]
            )
        if np.isinf(float(t1)):
            # Terminal segment: precompute up to a finite horizon and clamp.
            tmax_env = os.environ.get("DEMESTATS_CCR_TERMINAL_TMAX", "")
            if tmax_env:
                tmax = float(tmax_env)
            else:
                # Default: a modest horizon in the internally-rescaled time units.
                # For typical scaling (Nref ~ 1e3-1e4), this corresponds to thousands
                # to tens of thousands of generations.
                tmax = max(1.0, 5.0 * float(t0))
            return _lift_expm(
                state=state,
                t0=t0,
                t1=t1,
                t1_eval=t0 + tmax,
                update_state=False,
                demo=demo,
                aux=aux,
            )
        return _lift_expm(state=state, t0=t0, t1=t1, demo=demo, aux=aux)
    raise NotImplementedError("CCR ODE lifting is not implemented for variable epochs.")
