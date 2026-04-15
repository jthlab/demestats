from itertools import combinations
from typing import Any, List, Mapping, Set, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap
from loguru import logger
from scipy.optimize import LinearConstraint, minimize

from demestats.fit.util import (
    _dict_to_vec,
    _vec_to_dict,
    _vec_to_dict_jax,
    create_inequalities,
    make_whitening_from_hessian,
    pullback_objective,
)
from demestats.icr import ICRCurve
from demestats.loglik.icr_loglik import icr_loglik

logger.disable("demestats")


def get_tree_from_positions_data_efficient(
    ts, num_samples=50, gap=150000, k=2, seed=5, option="random"
):
    """
    Extract coalescence-time data from trees sampled at regularly spaced genome
    positions, together with the corresponding population sampling configurations.
    This function will return an error if there's missing data (i.e. where a genomic position doesn't
    have a local tree).

    Parameters
    ----------
        ts : tskit.TreeSequence
            The input tree sequence from which local trees are sampled.
        num_samples : int, optional
            The number of sampling replicates to generate when
            ``option="random"``. Default is ``50``.
        gap : int or float, optional
            The spacing between successive genomic positions at which trees are
            queried. Default is ``150000``.
        k : int, optional
            The number of sampled nodes. Default is ``2``.
        seed : int, optional
            The random seed used to initialize JAX-based sampling. Default is ``5``.
        option : {"random", "all"}, optional
            Strategy used to generate samples. With ``"random"``, ``num_samples``
            random samples of size ``k`` are drawn without replacement. With
            ``"all"``, every possible combination of ``k`` samples is used. If the total number
            of haploids (N) in the tree is large, do not use "all" as the number of total sampling
            configurations is N choose k.

    Returns
    -------
        tuple
            A pair ``(data, cfg_list)`` where:

            ``data`` is a JAX array containing the extracted coalescence times for
            each sample across queried genomic positions.

            ``cfg_list`` is a list of dictionaries giving the per-population sample
            counts.

    Notes
    -----
    The queried positions are constructed by drawing a random starting position in
    the first interval of length ``gap`` and then stepping across the genome in
    increments of ``gap``. For each queried position and each sampled tree
    sequence, the function records the time to first coalescence across the sampled
    nodes.

    When ``option="random"``, repeated random samples are drawn. When
    ``option="all"``, all N choose ``k`` sample combinations are enumerated, which may
    be expensive for large ``ts.num_samples``.

    ::
        data, cfgs = get_tree_from_positions_data_efficient(
            ts,
            num_samples=100,
            gap=100000,
            k=2,
            seed=42,
            option="random",
        )

        data, cfgs = get_tree_from_positions_data_efficient(
            ts,
            gap=50000,
            k=2,
            option="all",
        )
    """
    key = jax.random.PRNGKey(seed)
    num_trees = jnp.floor(ts.sequence_length / gap)
    start_position = jax.random.randint(key, (1,), 1, gap + 1)
    additive = jnp.arange(num_trees) * gap
    position_index = start_position + additive
    key, subkey = jr.split(key)
    data_list = []
    cfg_list = []
    pop_cfg = {
        ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()
    }

    if option == "random":
        for i in range(num_samples):
            tmp_data_list = []
            samples = jax.random.choice(
                subkey, ts.num_samples, shape=(k,), replace=False
            )
            key, subkey = jr.split(key)
            subsample_ts = ts.simplify(samples)

            tmp_pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

            for j in samples:
                tmp_pop_cfg[
                    ts.population(ts.node(j.item(0)).population).metadata["name"]
                ] += 1

            cfg_list.append(tmp_pop_cfg)

            for pos in position_index:
                tree = subsample_ts.at(pos)
                tmp_data_list.append(
                    min(
                        [
                            tree.time(tree.parent(node))
                            for node in subsample_ts.samples()
                        ]
                    )
                )

            data_list.append(tmp_data_list)

    if option == "all":
        all_configs = list(combinations(jnp.arange(ts.num_samples), k))

        for config in all_configs:
            subsample_ts = ts.simplify(config)
            tmp_data_list = []

            tmp_pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

            for j in config:
                tmp_pop_cfg[
                    ts.population(ts.node(j.item(0)).population).metadata["name"]
                ] += 1

            cfg_list.append(tmp_pop_cfg)

            for pos in position_index:
                tree = subsample_ts.at(pos)
                tmp_data_list.append(
                    min(
                        [
                            tree.time(tree.parent(node))
                            for node in subsample_ts.samples()
                        ]
                    )
                )

            data_list.append(tmp_data_list)

    return jnp.array(data_list), cfg_list


def process_data(cfg_list):
    """
    Convert a list of dictionary sampling configurations into a vectorized form for comptability with JAX.

    Parameters
    ----------
        cfg_list : sequence of dict
            A sequence of dictionaries where each dictionary maps deme names to the
            number of sampled haploids in that configuration.

    Returns
    -------
        tuple
            A pair ``(cfg_mat, deme_names)`` where:

            ``cfg_mat`` is a JAX integer array of shape ``(num_samples, D)``
            containing the sampling counts for each configuration and deme.

            ``deme_names`` is the ordered collection of deme names corresponding to
            the columns of ``cfg_mat``.

    Notes
    -----
    The deme ordering is taken from the keys of the first configuration in
    ``cfg_list`` and is used consistently for every row in the output matrix.
    If a deme is missing from a later configuration, its count is filled with ``0``.

    This function is used for converting a list-based representation of sampling
    configurations into a compact array form suitable for downstream numerical
    computation.

    ::
        cfg_mat, deme_names = process_data([
            {"P0": 2, "P1": 0},
            {"P0": 1, "P1": 1},
            {"P0": 0, "P1": 2},
        ])
    """
    num_samples = len(cfg_list)

    deme_names = cfg_list[0].keys()
    D = len(deme_names)
    cfg_mat = jnp.zeros((num_samples, D), dtype=jnp.int32)
    for i, cfg in enumerate(cfg_list):
        for j, n in enumerate(deme_names):
            cfg_mat = cfg_mat.at[i, j].set(cfg.get(n, 0))
    return cfg_mat, deme_names


Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _compute_icr_likelihood(vec, args_nonstatic, args_static):
    path_order, data, cfg_mat = args_nonstatic
    icr_call, deme_names = args_static
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("param: {vec}", vec=vec)
    batched_loglik = vmap(icr_loglik, in_axes=(0, 0, None, None, None))(
        data, cfg_mat, params, icr_call, deme_names
    )
    # jax.debug.print("batched_loglik: {}", batched_loglik)
    loss = -jnp.sum(batched_loglik)
    jax.debug.print("Loss: {loss}", loss=loss)
    return loss


def neg_loglik(vec, g, preconditioner_nonstatic, args_nonstatic, lb, ub):
    if jnp.any(vec >= ub):
        return jnp.inf, jnp.full_like(vec, 1e10)

    if jnp.any(vec <= lb):
        return jnp.inf, jnp.full_like(vec, -1e10)

    return g(vec, preconditioner_nonstatic, args_nonstatic)


def fit(
    demo,
    paths: Params,
    data,
    cfg_list,
    cons,
    lb,
    ub,
    k,
    *,
    method: str = "trust-constr",
    gtol: float = 1e-8,
    xtol: float = 1e-8,  # default 1e-8
    maxiter: int = 1000,  # default 1000
    barrier_tol: float = 1e-8,
):
    """
    Fit demographic model parameters using ICR likelihood optimization.

    Parameters
    ----------
    demo : demes.Graph
        ``demes`` model graph.
    paths : Params
        Parameter paths to optimize. Each path specifies a demographic
        parameter in the model.
    data : array_like
        array containing the extracted coalescence times for
        each subsample across queried genomic positions
    cfg_list : list of dictionaries
        list of dictionaries giving the per-population sample
        counts associated with each sample
    cons : dict
        Dictionary containing equality and inequality constraints.
        Expected keys: 'eq' for (Aeq, beq) equality constraints Aeq@x = beq,
        and 'ineq' for (G, h) inequality constraints G@x <= h.
    lb : array_like
        Lower bounds for parameters.
    ub : array_like
        Upper bounds for parameters.
    k : int
        Sample size 
    method : str, optional
        Optimization method (default: "trust-constr").
    gtol : float, optional
        Gradient tolerance for convergence (default: 1e-5).
    xtol : float, optional
        Parameter tolerance for convergence (default: 1e-5).
    maxiter : int, optional
        Maximum number of iterations (default: 1000).
    barrier_tol : float, optional
        Barrier tolerance for interior-point methods (default: 1e-5).

    Returns
    -------
    tuple
        (params_opt, opt_value, x_opt) where:
        - params_opt: Dictionary of optimized parameters
        - opt_value: Optimal negative log-likelihood value
        - x_opt: Optimized parameter vector

    Notes
    -----
    This function implements a sophisticated optimization pipeline:
    1. Parameter space transformation using Hessian-based whitening
    2. Constraint handling with equality and inequality constraints
    3. Optional random projections for computational efficiency
    4. Boundary enforcement with penalty gradients

    The optimization is performed in a transformed space where the Hessian
    is approximately identity, improving convergence rates.
    """
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    data = jnp.array(data)
    x0 = jnp.array(x0)
    lb = jnp.array(lb)
    ub = jnp.array(ub)

    cfg_mat, deme_names = process_data(cfg_list)
    icr = ICRCurve(demo=demo, k=k)
    icr_call = jax.jit(icr.__call__)

    args_nonstatic = (path_order, data, cfg_mat)
    args_static = (icr_call, deme_names)
    L, LinvT = make_whitening_from_hessian(
        _compute_icr_likelihood, x0, args_nonstatic, args_static
    )
    preconditioner_nonstatic = (x0, LinvT)
    g = pullback_objective(_compute_icr_likelihood, args_static)
    y0 = np.zeros_like(x0)

    lb_tr = L.T @ (lb - x0)
    ub_tr = L.T @ (ub - x0)

    linear_constraints: list[LinearConstraint] = []

    Aeq, beq = cons["eq"]
    A_tilde = Aeq @ LinvT
    b_tilde = beq - Aeq @ x0
    if Aeq.size:
        linear_constraints.append(LinearConstraint(A_tilde, b_tilde, b_tilde))

    G, h = cons["ineq"]
    if G.size:
        linear_constraints.append(create_inequalities(G, h, LinvT, x0))

    res = minimize(
        fun=neg_loglik,
        x0=y0,
        jac=True,
        args=(g, preconditioner_nonstatic, args_nonstatic, lb_tr, ub_tr),
        method=method,
        constraints=linear_constraints,
        options={
            "gtol": gtol,
            "xtol": xtol,
            "maxiter": maxiter,
            "barrier_tol": barrier_tol,
        },
    )

    x_opt = np.array(x0) + LinvT @ res.x
    print("optimal value: ")
    print(x_opt)
    print(res)

    return _vec_to_dict(jnp.asarray(x_opt), path_order), res.fun, x_opt
