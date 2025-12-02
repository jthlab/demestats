from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from demesinfer.fit.util import _dict_to_vec, _vec_to_dict_jax, _vec_to_dict, create_inequalities, make_whitening_from_hessian, pullback_objective
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import LinearConstraint, minimize
from demesinfer.iicr import IICRCurve
from jax import vmap
from loguru import logger
logger.disable("demesinfer")
import jax.random as jr
from itertools import combinations

def get_tree_from_positions_data_efficient(ts, num_samples=50, gap=150000, k=2, seed=5, option="random"):
    key = jax.random.PRNGKey(seed)
    num_trees = jnp.floor(ts.sequence_length / gap)
    start_position = jax.random.randint(key, (1,), 1, gap+1)
    additive = jnp.arange(num_trees) * gap
    position_index = start_position + additive
    key, subkey = jr.split(key)
    data_list = []
    cfg_list = []
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    
    if option == "random":
        for i in range(num_samples):
            tmp_data_list = []
            samples = jax.random.choice(subkey, ts.num_samples, shape=(k,), replace=False)
            key, subkey = jr.split(key)
            subsample_ts = ts.simplify(samples)

            tmp_pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

            for j in samples:
               tmp_pop_cfg[ts.population(ts.node(j.item(0)).population).metadata["name"]] += 1

            cfg_list.append(tmp_pop_cfg)
    
            for pos in position_index:
                tree = subsample_ts.at(pos)
                tmp_data_list.append(min([tree.time(tree.parent(node)) for node in subsample_ts.samples()]))

            data_list.append(tmp_data_list)

    if option == "all":
        all_configs = list(combinations(jnp.arange(ts.num_samples), k)) 

        for config in all_configs:
            subsample_ts = ts.simplify(config)
            tmp_data_list = []

            tmp_pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

            for j in config:
               tmp_pop_cfg[ts.population(ts.node(j.item(0)).population).metadata["name"]] += 1

            cfg_list.append(tmp_pop_cfg)

            for pos in position_index:
                tree = subsample_ts.at(pos)
                tmp_data_list.append(min([tree.time(tree.parent(node)) for node in subsample_ts.samples()]))

            data_list.append(tmp_data_list)
    
    return jnp.array(data_list), cfg_list
    
def process_data(cfg_list):
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

def compute_loglik(time, sample_config, params, iicr_call, deme_names):
   ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
   result = iicr_call(params=params, t=time, num_samples=ns)
   # jax.debug.print("result: {}", result)
   return jnp.sum(jnp.log(result["c"]) + result["log_s"])

def _compute_mrpast_likelihood(vec, args_nonstatic, args_static):
    path_order, data, cfg_mat = args_nonstatic
    iicr_call, deme_names = args_static
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("param: {vec}", vec=vec)
    batched_loglik = vmap(compute_loglik, in_axes=(0, 0, None, None, None))(data, cfg_mat, params, iicr_call, deme_names)
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
    xtol: float = 1e-8, #default 1e-8
    maxiter: int = 1000, #default 1000
    barrier_tol: float = 1e-8,
):
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)
    data = jnp.array(data)
    x0 = jnp.array(x0)
    lb = jnp.array(lb)
    ub = jnp.array(ub)

    cfg_mat, deme_names = process_data(cfg_list)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    args_nonstatic = (path_order, data, cfg_mat)
    args_static = (iicr_call, deme_names)
    L, LinvT = make_whitening_from_hessian(_compute_mrpast_likelihood, x0, args_nonstatic, args_static)
    preconditioner_nonstatic = (x0, LinvT)
    g = pullback_objective(_compute_mrpast_likelihood, args_static)
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
        args = (g, preconditioner_nonstatic, args_nonstatic, lb_tr, ub_tr),
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

    return _vec_to_dict(jnp.asarray(res.x), path_order), res.fun, x_opt