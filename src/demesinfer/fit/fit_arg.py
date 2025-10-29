from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize
from jax import vmap
from demesinfer.iicr import IICRCurve
from jax.scipy.special import xlogy
import numpy as np

from loguru import logger
logger.disable("demesinfer")

import diffrax as dfx
from jaxtyping import Array, Float, Scalar, ScalarLike

from demesinfer.fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, create_inequalities, make_whitening_from_hessian, pullback_objective, process_arg_data, reformat_data
Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]
from demesinfer.bounded_solver import BoundedSolver

def loglik_ode(params, sample_config, deme_names, iicr_call, r: ScalarLike, times: Float[Array, "intervals 1"]) -> Scalar:
    # times = data.T
    # i = times.argsort()
    # sorted_times = times[i]
    ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
    single_config_func = iicr_call(params=params, num_samples=ns)
    dictionary = jax.vmap(single_config_func, in_axes=(0))(times)

    def f(t, y, _):
        c = jnp.clip(single_config_func(t)["c"], a_min=1e-21, a_max=None)
        A = jnp.array([[-r, r, 0.0], [c, -2 * c, c], [0.0, 0.0, 0.0]])
        return A.T @ y

    y0 = jnp.array([1.0, 0.0, 0.0])
    
    def oob_fn(y):
        return jnp.any(y < 0.0) | jnp.any(y > 1.0)

    solver = BoundedSolver(oob_fn=oob_fn)
    
    term = dfx.ODETerm(f)
    ssc = dfx.PIDController(rtol=1e-6, atol=1e-6, jump_ts=single_config_func.jump_ts)
    # ssc = dfx.PIDController(rtol=1e-6, atol=1e-6)
    T = times.max()
    sol = dfx.diffeqsolve(
        term,
        solver,
        0.0,
        T,
        dt0=0.001,
        y0=y0,
        stepsize_controller=ssc,
        saveat=dfx.SaveAt(dense=True),
    )

    # invert the sorting so that cscs matches times
    # i_inv = i.argsort()
    # cscs = sol.ys[i_inv]
    cscs = jax.vmap(sol.evaluate)(times)
    return cscs, jnp.clip(dictionary["c"], a_min=1e-21, a_max=None), dictionary["log_s"]

def compute_loglik(data, cscs, max_index, c_map, log_s):
    times, spans = data.T
    @vmap
    def p(t0, csc0, t1, csc1, span, c_rate_t1, log_s_t0, log_s_t1):
        p_nr_t0, p_float_t0, p_coal_t0 = csc0
        p_nr_t1, p_float_t1, p_coal_t1 = csc1
        # no recomb for first span - 1 positions
        r1 = xlogy(span - 1, p_nr_t0)
        # coalescence at t1
        r2 = jnp.log(c_rate_t1)
        # back-coalescence process up to t1, depends to t0 >< t1
        r3 = jnp.where(
            t0 < t1, jnp.log(p_float_t0) + log_s_t1 - log_s_t0, jnp.log(p_float_t1) # t0 < t1, jnp.log(p_float_t0) - eta.R(t0, t1), jnp.log(p_float_t1)
        )
        return r1 + r2 + r3

    # ll = p(times[:-1], cscs[:-1], times[1:], cscs[1:], spans[:-1], c_map[1:], log_s[:-1], log_s[1:])
    # ll = jnp.dot(ll, jnp.arange(len(times[:-1])) < max_index)
    
    # # for the last position, we only know span was at least as long
    # ll += xlogy(spans[max_index], cscs[max_index, 0])
    ll = p(times[:-1], cscs[:-1], times[1:], cscs[1:], spans[:-1], c_map[1:], log_s[:-1], log_s[1:])
    ll = jnp.sum(ll, where=spans[1:] > 0.0)
    return ll

def _compute_arg_likelihood(vec, path_order, unique_cfg, deme_names, iicr_call, rho, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices):
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("Params: {params}", params=vec)
    
    all_cscs, c_map, log_s = jax.vmap(loglik_ode, in_axes=(None, 0, None, None, None, 0))(params, unique_cfg, deme_names, iicr_call, rho, unique_times)
    extracted = all_cscs[group_membership, associated_indices]
    final_cscs_flat = extracted.reshape(batch_size, chunking_length, 3)
    log_s = log_s[group_membership, associated_indices].reshape(batch_size, chunking_length)
    c_map = c_map[group_membership, associated_indices].reshape(batch_size, chunking_length)
    batched_loglik = vmap(compute_loglik, in_axes=(0, 0, 0, 0, 0))(rearranged_data, final_cscs_flat, new_max_indices, c_map, log_s)
    
    loss = -jnp.sum(batched_loglik)
    jax.debug.print("Loss: {loss}", loss=loss)
    return loss

def neg_loglik(vec, g, lb, ub):
    if jnp.any(vec > ub) or jnp.any(vec < lb):
        return jnp.inf, jnp.full_like(vec, jnp.inf)

    return g(vec)

def fit(
    demo,
    paths: Params,
    data_list,
    cfg_list,
    cons,
    lb,
    ub,
    *,
    method: str = "trust-constr",
    recombination_rate = 1e-8,
    mutation_rate = 1e-8,
    window_size = 100,
    seed: float = 5, 
    gtol: float = 1e-8,
    xtol: float = 1e-8, #default 1e-8
    maxiter: int = 1000, #default 1000
    barrier_tol: float = 1e-8,
):
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    x0 = jnp.array(x0)
    lb = jnp.array(lb)
    ub = jnp.array(ub)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(data_list, cfg_list)
    chunking_length = data_pad.shape[1]
    unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    num_samples = len(max_indices)
    rho = recombination_rate = 1e-8
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = iicr.curve
        
    args = (path_order, unique_cfg, deme_names, iicr_call, rho, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices)
    L, LinvT = make_whitening_from_hessian(_compute_arg_likelihood, x0, *args)
    g = pullback_objective(_compute_arg_likelihood, x0, LinvT, *args)
    y0 = np.zeros_like(x0)

    lb_tr = L.T @ (lb - x0)
    ub_tr = L.T @ (ub - x0)

    linear_constraints: list[LinearConstraint] = []
    
    Aeq, beq = cons["eq"]
    A_tilde = Aeq @ LinvT
    b_tilde = beq - Aeq@x0
    if Aeq.size:
        linear_constraints.append(LinearConstraint(A_tilde, b_tilde, b_tilde))

    G, h = cons["ineq"]
    if G.size:
        linear_constraints.append(create_inequalities(G, h, LinvT, x0, size=len(paths)))
    
    res = minimize(
        fun=neg_loglik,
        x0=y0,
        jac=True,
        args = (g, lb_tr, ub_tr),
        method=method,
        constraints=linear_constraints,
        options={
            'gtol': gtol,
            'xtol': xtol, 
            'maxiter': maxiter,
            'barrier_tol': barrier_tol,
        }
    )

    x_opt = np.array(x0) + LinvT @ res.x
    print("optimal value: ")
    print(x_opt)
    print(res)

    return _vec_to_dict(jnp.asarray(res.x), path_order), res