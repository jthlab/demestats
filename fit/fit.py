# Example implementation of a fit function for parameter inference.
# This is intended for tutorial use only. We do not take responsibility for any bugs or issues in this code.

from __future__ import annotations
from typing import Mapping, Sequence, Tuple, List, Dict, Any, Optional, Set

import jax
import jax.numpy as jnp
import msprime as msp
from scipy.optimize import minimize, LinearConstraint

from demesinfer.constr import constraints_for, EventTree
from demesinfer.iicr import IICRCurve
from demesinfer.coal_rate import PiecewiseConstant
from demesinfer.loglik import loglik

Path  = Tuple[Any, ...]
Var   = Path | Set[Path]             
Params = Mapping[Var, float]


def _dict_to_vec(d: Params, keys: Sequence[Var]) -> jnp.ndarray:
    return jnp.asarray([d[k] for k in keys], dtype=jnp.float64)

def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    return {k: v[i] for i, k in enumerate(keys)}

def _vec_to_dict(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, float]:
    return {k: float(v[i]) for i, k in enumerate(keys)}


def _compile(ts):
    data, cfg = [], []
    ids = list(range(ts.num_samples))
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            spans, curr_t, curr_L = [], None, 0.0
            for tree in ts.trees():
                L = tree.interval.right - tree.interval.left
                t = tree.tmrca(a, b)
                if curr_t is None or t != curr_t:
                    if curr_t is not None:
                        spans.append([curr_t, curr_L])
                    curr_t, curr_L = t, L
                else:
                    curr_L += L
            spans.append([curr_t, curr_L])
            data.append(jnp.asarray(spans, dtype=jnp.float64))

            pop_cfg[ts.population(ts.node(a).population).metadata["name"]] += 1
            pop_cfg[ts.population(ts.node(b).population).metadata["name"]] += 1
            cfg.append(
                {
                    ts.population(ts.node(a).population).metadata["name"]: 1,
                    ts.population(ts.node(b).population).metadata["name"]: 1,
                }
            )
    return data, cfg


def fit(
    demo,
    paths: Params,
    *,
    k: int = 2,
    n_samples: int = 10,
    t_min: float = 1e-8,
    t_max: float,
    num_t: int = 1000,
    method: str = "trust-constr",
    options: Optional[dict] = None,
    recombination_rate: float = 1e-8,
    sequence_length: float = 1e7,
    mutation_rate: float = 1e-8,
    seed: int = 1,
):
    msp_demo = msp.Demography.from_demes(demo)
    deme_names = [d.name for d in demo.demes]
    samples = {d: n_samples for d in deme_names[1:]}
    ts = msp.sim_mutations(
        msp.sim_ancestry(
            samples=samples,
            demography=msp_demo,
            recombination_rate=recombination_rate,
            sequence_length=sequence_length,
            random_seed=seed,
        ),
        rate=mutation_rate,
        random_seed=seed + 1,
    )

    data_list, cfg_list = _compile(ts)
    path_order: List[Var] = list(paths)
    x0 = _dict_to_vec(paths, path_order)

    et = EventTree(demo)

    cons = constraints_for(et, *path_order)
    linear_constraints: list[LinearConstraint] = []

    Aeq, beq = cons["eq"]
    if Aeq.size:
        linear_constraints.append(LinearConstraint(Aeq, beq, beq))

    G, h = cons["ineq"]
    if G.size:
        lower = -jnp.inf * jnp.ones_like(h)
        linear_constraints.append(LinearConstraint(G, lower, h))

    lens = jnp.array([d.shape[0] for d in data_list], dtype=jnp.int32)
    Lmax = int(lens.max())
    Npairs = len(data_list)

    data_pad = jnp.full((Npairs, Lmax, 2), jnp.array([1.0, 0.0]), dtype=jnp.float64)
    for i, d in enumerate(data_list):
        data_pad = data_pad.at[i, : d.shape[0], :].set(d)

    D = len(deme_names)
    cfg_mat = jnp.zeros((Npairs, D), dtype=jnp.int32)
    for i, cfg in enumerate(cfg_list):
        for j, n in enumerate(deme_names):
            cfg_mat = cfg_mat.at[i, j].set(cfg.get(n, 0))

    cfg_keys, cfg_idx = jnp.unique(cfg_mat, axis=0, return_inverse=True)

    t_breaks = jnp.linspace(t_min, t_max, num_t + 1)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)
    
    #@jax.jit
    @jax.value_and_grad
    def neg_loglik(vec):
        params = _vec_to_dict_jax(vec, path_order)
        def build_eta(row):
            ns = {name: row[i] for i, name in enumerate(deme_names)}
            return iicr_call(params=params, t=t_breaks[:-1], num_samples=ns)["c"]
        eta_stack = jax.vmap(build_eta)(cfg_keys)
        
        total_ll = 0.0
        for i in range(Npairs):
            data = data_list[i]
            idx = int(cfg_idx[i]) 
            eta = PiecewiseConstant(c=eta_stack[idx], t=t_breaks[:-1])
            total_ll += loglik(eta, rho, data)
            
        return -total_ll / Npairs
    
    res = minimize(
        fun=lambda x: float(neg_loglik(x)[0]),
        x0=jnp.asarray(x0),
        jac=lambda x: jnp.asarray(neg_loglik(x)[1], dtype=float),
        method=method,
        constraints=linear_constraints,
        options=options,
    )

    return _vec_to_dict(jnp.asarray(res.x), path_order)
