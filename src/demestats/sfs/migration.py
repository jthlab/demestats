from collections import OrderedDict
from collections.abc import Sequence
from functools import partial

import diffrax as dfx
import jax
import jax.experimental.sparse as jesp
import jax.numpy as jnp
import scipy.sparse as sps
from beartype.typing import Callable
from jaxtyping import ArrayLike, ScalarLike
from loguru import logger
from penzai import pz
from scipy.sparse.linalg._expm_multiply import _expm_multiply_simple

from demestats.pexp import PExp

from .momints import _drift, _migration, _mutation


def _dense_expmv(A, v, t):
    result_shape = jax.ShapeDtypeStruct(v.shape, v.dtype)
    return jax.pure_callback(_expm_multiply_simple, result_shape, A, v, t)


def lift_cm_aux(axes: OrderedDict, migration_pairs: Sequence[tuple[str, str]]) -> dict:
    # compute the transition matrices for the dimensions involved in the lift. this function doesn't know about the
    # other dimensions that are not being lifted.
    tm = {}
    pops = list(axes.keys())
    tm["drift"] = {pop: _drift(axes[pop] - 1) for pop in pops}
    tm["mut"] = {pop: _mutation(axes[pop] - 1) for pop in pops}
    tm["mig"] = {
        (p1, p2): _migration((axes[p1] - 1, axes[p2] - 1)) for p1, p2 in migration_pairs
    }

    # convert sparse matrices from scipy to JAX format
    def f(A):
        if isinstance(A, sps.sparray):
            ret = jesp.BCOO.from_scipy_sparse(A).sort_indices()
            ret.unique_indices = True
            return A.todense()
            return ret
        return A

    tm = jax.tree.map(f, tm)
    tm["axes"] = axes
    return tm


def _e0_like(pl):
    return jnp.zeros(pl.size).at[0].set(1.0).reshape(pl.shape)


def lift_cm(
    pl: pz.nx.NamedArray,
    t0: ScalarLike,
    t1: ScalarLike,
    etas: dict[str, PExp],
    mu: Callable,
    demo: dict,
    aux: dict,
    etbl: bool,
):
    if False:
        logger.debug("using sparse matrix exponentiation for {}", aux)
        f = _lift_cm_exp
    else:
        logger.debug("using diffeq solver for lift")
        f = _lift_cm_exp
    return f(pl, t0, t1, etas, mu, demo, aux, etbl)


def tensormul(
    tens: list[dict[str, ArrayLike]], pl: pz.nx.NamedArray, tr: bool = False
) -> pz.nx.NamedArray:
    "multiply a named array by a list of tensors, each of which is a dict mapping some axes to arrays"
    ret = 0.0
    for T in tens:
        pops = list(T.keys())
        mats = list(T.values())
        K = len(pops)
        args = []
        for i, M in enumerate(mats):
            ax = [K + i, i]
            if tr:
                ax = ax[::-1]
            args.extend([M, ax])
        A_axes = list(range(K))
        out_axes = list(range(K, 2 * K))

        @pz.nx.nmap
        def f(A):
            return jnp.einsum(A, A_axes, *args, out_axes)

        ret += f(pl.untag(*pops)).tag(*pops)
    return ret


def ode(s, y, args, etbl):
    (t0, t1, etas, mu, aux, pops) = args
    y = tuple([pz.nx.wrap(a, *pops) for a in y])

    def Q_drift(s):
        ret = []
        for pop in y[0].named_axes:
            c = 1 / (4 * etas[pop](s))
            ret.append({pop: aux["drift"][pop] * c})
        return ret

    def Q_mig(s):
        ret = []
        for (p1, p2), (u1, u2) in aux["mig"].items():
            m = mu(p1, p2, s)
            ret.extend(
                [
                    {p1: m * u1[0], p2: u1[1]},
                    {p2: m * u2[1]},
                ]
            )
        return ret

    Q_mut = [{k: v} for k, v in aux["mut"].items() if k in pops]

    if not etbl:
        Qd = Q_drift(s)
        Qm = Q_mig(s)

        # backwards in time: t0 -> t1, solve for partial likelihood
        Q0 = Qd + Qm
        ret = (tensormul(Q0, y[0], tr=True),)
    else:
        # forwards in time: t1 -> t0, solve for expected branch length
        sp = t1 + t0 - s
        Q1 = Q_mig(sp) + Q_drift(sp)
        r0 = tensormul(Q1, y[0]) + tensormul(Q_mut, y[1])
        r1 = tensormul(Q1, y[1])
        ret = (r0, r1)
    return tuple([a.unwrap(*pops) for a in ret])


def _lift_cm_exp(
    pl: pz.nx.NamedArray,
    t0: ScalarLike,
    t1: ScalarLike,
    etas: dict[str, Callable],
    mu: Callable,
    demo: dict,
    aux: dict,
    etbl: bool,
):
    # population sizes are changing, so we have to use a differential
    # equation solver
    dims = [pl.named_axes[pop] for tup in aux["mig"] for pop in tup]
    assert all(a >= 5 for a in dims), (
        "dimensions too small for migration, require n >= 4"
    )
    solver = dfx.Tsit5()
    # solver = dfx.Kvaerno3()
    eta_ts = jnp.concatenate([eta.t[:-1] for eta in etas.values()])
    mu_ts = jnp.array(
        [m.get(x, t0) for m in demo["migrations"] for x in ("start_time", "end_time")]
    )
    jump_ts = jnp.concatenate([eta_ts, mu_ts])
    jump_ts = jnp.sort(jump_ts)
    ssc = dfx.PIDController(jump_ts=jump_ts, rtol=1e-6, atol=1e-6)

    # compute d/dtheta x(t,theta)|{theta=0} using the forward sensitivity method.
    # we have x'(t, theta) = Q(t, theta) @ x(t, theta) and therefore
    # d/dtheta x'(t, theta) = dQ/dtheta @ x + Q @ (dx/dtheta)
    # = (Q_mut @ x) + Q(t) @ (dx/dtheta)
    # d/dt dx(t,theta)/dtheta d/dtheta x'(t,theta) = dQ/dtheta @ x + Q @ (dx/dtheta)
    #   = (Q_mut @ x) + Q(t) @ (dx/dtheta)
    # dF/dtheta = d(Q @ x)/dtheta = (Q_mut @ x)
    # the initial condition is d/dtheta(x(0, theta)) = 0.; x(0,theta) = e0
    # for computing branch length, we only need to track the populations that are involved in the migration
    z = pz.nx.zeros(pl.named_axes)
    e0 = z.at[{k: 0 for k in z.named_axes}].set(1.0)
    if etbl:
        y0 = (z, e0)
    else:
        y0 = (pl,)

    term = dfx.ODETerm(partial(ode, etbl=etbl))

    # we can't pass NamedArrays into and out of the ODE solver, so we have to convert
    pops = tuple(pl.named_axes.keys())
    y0 = tuple([a.unwrap(*pops) for a in y0])

    # catch errors early
    args = (t0, t1, etas, mu, aux, pops)
    res = ode(0.0, y0, args, etbl)

    res = dfx.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=(t1 - t0) / 100.0,
        # dt0=None,
        y0=y0,
        args=args,
        stepsize_controller=ssc,
        # max_steps=4096,
        # max_steps=65536,
        # adjoint=dfx.RecursiveCheckpointAdjoint(checkpoints=10),
        # adjoint=dfx.BacksolveAdjoint(),
        # adjoint=dfx.DirectAdjoint(),
    )
    if etbl:
        etbl = res.ys[0][0]
        for x in (0, -1):
            etbl = etbl.at[(x,) * etbl.ndim].set(0.0)
        return etbl
    else:
        plp = res.ys[0][0]
        return plp
