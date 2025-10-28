from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from demesinfer.fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, create_inequalities, make_whitening_from_hessian, pullback_objective
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize
from demesinfer.iicr import IICRCurve
from demesinfer.fit.util import process_data
from jax import vmap
from phlashlib.iicr import PiecewiseConstant
from phlashlib.loglik import loglik

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def process_base_model(deme_names, cfg, iicr, num_timepoints):
    curve = iicr.curve(num_samples=dict(zip(deme_names, cfg)))
    timepoints = jax.vmap(curve.quantile)(jnp.linspace(0, 1, num_timepoints)[1:-1])
    timepoints = jnp.insert(timepoints, 0, 0.0)
    return timepoints

def compute_loglik(c_map, c_index, data, times, theta, rho):
    c = c_map[c_index]
    t = times[c_index]
    eta = PiecewiseConstant(c=c, t=t)
    return loglik(data, eta, t, theta, rho)

def _compute_phlashlib_likelihood(vec, path_order, times, deme_names, unique_cfg, matching_indices, het_matrix, iicr_call, theta, rho):
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("Params: {params}", params=vec)
    
    c_map = jax.vmap(lambda cfg, time: iicr_call(params=params, t=time, num_samples=dict(zip(deme_names, cfg)))["c"])(
        unique_cfg, times
    )
    
    batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0, None, None, None))(c_map, matching_indices, het_matrix, times, theta, rho)
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
    het_matrix,
    cfg_list,
    cons,
    lb,
    ub,
    *,
    method: str = "trust-constr",
    num_timepoints = 33,
    recombination_rate = 1e-8,
    mutation_rate = 1e-8,
    window_size = 100,
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
    het_matrix = jnp.array(het_matrix)

    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    unique_cfg = jnp.array(unique_cfg)

    rho = recombination_rate * window_size
    theta = mutation_rate * window_size
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = jax.jit(iicr.__call__)
    
    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(deme_names, unique_cfg, iicr, num_timepoints)
    
    args = (path_order, times, deme_names, unique_cfg, matching_indices, het_matrix, iicr_call, theta, rho)
    L, LinvT = make_whitening_from_hessian(_compute_phlashlib_likelihood, x0, *args)
    g = pullback_objective(_compute_phlashlib_likelihood, x0, LinvT, *args)
    y0 = np.zeros_like(x0)

    lb_tr = L.T @ (lb - x0)
    ub_tr = L.T @ (ub - x0)

    print(lb_tr)
    print(ub_tr)

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