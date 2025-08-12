# Example implementation of a fit function for parameter inference.
# This is intended for tutorial use only. We do not take responsibility for any bugs or issues in this code.

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import jax
import jax.numpy as jnp
import msprime as msp
from scipy.optimize import LinearConstraint, minimize
import jax.random as jr
from jax import vmap, lax 

from demesinfer.coal_rate import PiecewiseConstant
from demesinfer.constr import EventTree, constraints_for
from demesinfer.iicr import IICRCurve
from demesinfer.loglik.arg import loglik

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _dict_to_vec(d: Params, keys: Sequence[Var]) -> jnp.ndarray:
    return jnp.asarray([d[k] for k in keys], dtype=jnp.float64)

def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    return {k: v[i] for i, k in enumerate(keys)}

def _vec_to_dict(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, float]:
    return {k: float(v[i]) for i, k in enumerate(keys)}

def compile(ts, subkey):
    # using a set to pull out all unique populations that the samples can possibly belong to
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    samples = jax.random.choice(subkey, ts.num_samples, shape=(2,), replace=False)
    a, b = samples[0].item(0), samples[1].item(0)
    spans = []
    curr_t = None
    curr_L = 0.0
    for tree in ts.trees():
        L = tree.interval.right - tree.interval.left
        t = tree.tmrca(a, b)
        if curr_t is None or t != curr_t:
            if curr_t is not None:
                spans.append([curr_t, curr_L])
            curr_t = t
            curr_L = L
        else:
            curr_L += L
    spans.append([curr_t, curr_L])
    data = jnp.asarray(spans, dtype=jnp.float64)
    pop_cfg[ts.population(ts.node(a).population).metadata["name"]] += 1
    pop_cfg[ts.population(ts.node(b).population).metadata["name"]] += 1
    return data, pop_cfg

def get_tmrca_data(ts, key, num_samples):
    data_list = []
    cfg_list = []
    max_indices = []
    for i in range(num_samples):
        key, subkey = jr.split(key)
        data, cfg = compile(ts, subkey)
        data_list.append(data)
        cfg_list.append(cfg)
        max_indices.append(data.shape[0] - 1)

    lens = jnp.array([d.shape[0] for d in data_list], dtype=jnp.int32)
    Lmax = int(lens.max())
    Npairs = len(data_list)
    data_pad = jnp.full((Npairs, Lmax, 2), jnp.array([1.0, 0.0]), dtype=jnp.float64)

    for i, d in enumerate(data_list):
        data_pad = data_pad.at[i, : d.shape[0], :].set(d)

    deme_names = cfg_list[0].keys()
    D = len(deme_names)
    cfg_mat = jnp.zeros((num_samples, D), dtype=jnp.int32)
    for i, cfg in enumerate(cfg_list):
        for j, n in enumerate(deme_names):
            cfg_mat = cfg_mat.at[i, j].set(cfg.get(n, 0))
    
    return data_pad, cfg_mat, deme_names, jnp.array(max_indices)

def plot_likelihood(demo, ts, paths, vec_values, recombination_rate=1e-8, seed=1, num_samples=20, t_min=1e-8, num_t=1000, k=2):
    import matplotlib.pyplot as plt

    key = jr.PRNGKey(seed)
    path_order: List[Var] = list(paths)
    data_pad, cfg_mat, deme_names, max_indices = get_tmrca_data(ts, key, num_samples)
    first_columns = data_pad[:, :, 0]
    # Compute global max (single float value)
    global_max = jnp.max(first_columns)
    t_breaks = jnp.linspace(t_min, global_max * 2, num_t)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    def compute_loglik(vec, sample_config, data, max_index):
        # Convert sample_config (array) to dictionary of population sizes
        ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
        
        params = _vec_to_dict_jax(vec, path_order)
        
        # Compute IICR and log-likelihood
        c = iicr_call(params=params, t=t_breaks, num_samples=ns)["c"]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(eta, rho, data, max_index)
    
    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        # Batched over cfg_mat and all_tmrca_spans 
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0, 0))(vec_array, cfg_mat, data_pad, max_indices)
        return -jnp.sum(batched_loglik) / num_samples  # Same as original neg_loglik

    # Outer vmap: Parallelize across vec_values
    # batched_neg_loglik = vmap(evaluate_at_vec)  # in_axes=0 is default

    # # 3. Compute all values (runs on GPU/TPU if available)
    # results = batched_neg_loglik(vec_values) 
    results = lax.map(evaluate_at_vec, vec_values)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def fit(
    demo,
    paths: Params,
    ts,
    *,
    k: int = 2,
    n_samples: int = 10,
    t_min: float = 1e-8,
    # t_max: float,
    num_t: int = 1000,
    method: str = "trust-constr",
    options: Optional[dict] = None,
    recombination_rate: float = 1e-8,
    sequence_length: float = 1e7,
    mutation_rate: float = 1e-8,
    seed: int = 1,
    num_samples = 20,
):
    key = jr.PRNGKey(seed)
    # msp_demo = msp.Demography.from_demes(demo)
    # deme_names = [d.name for d in demo.demes]
    # samples = {d: n_samples for d in deme_names[1:]}
    # ts = msp.sim_mutations(
    #     msp.sim_ancestry(
    #         samples=samples,
    #         demography=msp_demo,
    #         recombination_rate=recombination_rate,
    #         sequence_length=sequence_length,
    #         random_seed=seed,
    #     ),
    #     rate=mutation_rate,
    #     random_seed=seed + 1,
    # )

    data_pad, cfg_mat, deme_names, max_indices = get_tmrca_data(ts, key, num_samples)

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

    first_columns = data_pad[:, :, 0]
    # Compute global max (single float value)
    global_max = jnp.max(first_columns)
    t_breaks = jnp.linspace(t_min, global_max * 2, num_t)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    def compute_loglik(vec, sample_config, data, max_index):
        # Convert sample_config (array) to dictionary of population sizes
        ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
        
        params = _vec_to_dict_jax(vec, path_order)
        
        # Compute IICR and log-likelihood
        c = iicr_call(params=params, t=t_breaks, num_samples=ns)["c"]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(eta, rho, data, max_index)
    
    @jax.value_and_grad
    def neg_loglik(vec):
        vec = vec
        batched_loglik = vmap(
        compute_loglik,
        in_axes=(None, 0, 0, 0))(vec, cfg_mat, data_pad, max_indices)
        
        likelihood = jnp.sum(batched_loglik)

        return -likelihood / num_samples

    res = minimize(
        fun=lambda x: float(neg_loglik(x)[0]),
        # fun=lambda x: float(neg_loglik(x)),
        x0=jnp.asarray(x0),
        jac=lambda x: jnp.asarray(neg_loglik(x)[1], dtype=float),
        method=method,
        # bounds = [(3000. / 5000., 7000. / 5000.)],
        constraints=linear_constraints,
    )

    return _vec_to_dict(jnp.asarray(res.x), path_order)
