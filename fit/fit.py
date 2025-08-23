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
from jax.scipy.special import xlogy
from demesinfer.sfs import ExpectedSFS

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def _dict_to_vec(d: Params, keys: Sequence[Var]) -> jnp.ndarray:
    return jnp.asarray([d[k] for k in keys], dtype=jnp.float64)

def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    return {k: v[i] for i, k in enumerate(keys)}

def _vec_to_dict(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, float]:
    return {k: float(v[i]) for i, k in enumerate(keys)}

def compile(ts, subkey, a=None, b=None):
    # using a set to pull out all unique populations that the samples can possibly belong to
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    if a == None and b == None:
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

def get_tmrca_data(ts, key=jax.random.PRNGKey(2), num_samples=200, option="random"):
    data_list = []
    cfg_list = []
    if option == "random":
        for i in range(num_samples):
            key, subkey = jr.split(key)
            data, cfg = compile(ts, subkey)
            data_list.append(data)
            cfg_list.append(cfg)
    elif option == "all":
        from itertools import combinations
        all_config = list(combinations(ts.samples(), 2))
        num_samples = len(all_config)
        for a, b in all_config:
            data, cfg = compile(ts, subkey, a, b)
            data_list.append(data)
            cfg_list.append(cfg)

    return data_list, cfg_list     

def process_data(data_list, cfg_list):
    max_indices = jnp.array([arr.shape[0]-1 for arr in data_list])
    num_samples = len(max_indices)
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

    unique_cfg = jnp.unique(cfg_mat, axis=0)

    # Find matching indices
    def find_matching_index(row, unique_arrays):
        matches = jnp.all(row == unique_arrays, axis=1)
        return jnp.where(matches)[0][0]

    # Vectorize over all rows in `arr`
    matching_indices = jnp.array([find_matching_index(row, unique_cfg) for row in cfg_mat])
    
    return data_pad, cfg_mat, deme_names, max_indices, unique_cfg, matching_indices

def plot_iicr_likelihood(demo, data_list, cfg_list, paths, vec_values, recombination_rate=1e-8, t_min=1e-8, num_t=2000, k=2):
    import matplotlib.pyplot as plt

    path_order: List[Var] = list(paths)
    data_pad, cfg_mat, deme_names, max_indices, unique_cfg, matching_indices = process_data(data_list, cfg_list)
    num_samples = len(max_indices)
    first_columns = data_pad[:, :, 0]
    # Compute global max (single float value)
    global_max = jnp.max(first_columns)
    print(global_max)
    t_breaks = jnp.insert(jnp.geomspace(t_min, global_max, num_t), 0, 0.0)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    def compute_loglik(c_map, c_index, data, max_index):
        c = c_map[c_index]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(eta, rho, data, max_index)
    
    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        params = _vec_to_dict_jax(vec_array, path_order)

        # def compute_c(sample_config):
        #     # Convert sample_config (array) to dictionary of population sizes
        #     ns = {name: sample_config[i] for i, name in enumerate(deme_names)}
            
        #     # Compute IICR and log-likelihood
        #     c = iicr_call(params=params, t=t_breaks, num_samples=ns)["c"]
        #     return c
        # c_map = vmap(compute_c, in_axes=(0))(unique_cfg)
        c_map = jax.vmap(lambda cfg: iicr_call(params=params, t=t_breaks, num_samples=dict(zip(deme_names, cfg)))["c"])(
            jnp.array(unique_cfg)
        )
        
        # Batched over cfg_mat and all_tmrca_spans 
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0, 0))(c_map, matching_indices, data_pad, max_indices)
        return -jnp.sum(batched_loglik) / num_samples  # Same as original neg_loglik

    # Outer vmap: Parallelize across vec_values
    # batched_neg_loglik = vmap(evaluate_at_vec)  # in_axes=0 is default

    # 3. Compute all values (runs on GPU/TPU if available)
    # results = batched_neg_loglik(vec_values) 
    results = lax.map(evaluate_at_vec, vec_values)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_sfs_likelihood(demo, paths, vec_values, afs, afs_samples, theta=None, sequence_length=None):
    import matplotlib.pyplot as plt

    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    def sfs_loglik(afs, esfs, sequence_length, theta):
        afs = afs.flatten()[1:-1]
        esfs = esfs.flatten()[1:-1]
        
        if theta:
            assert(sequence_length)
            tmp = esfs * sequence_length * theta
            return jnp.sum(-tmp + xlogy(afs, tmp))
        else:
            return jnp.sum(xlogy(afs, esfs/esfs.sum()))
    
    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        params = _vec_to_dict_jax(vec_array, path_order)
        e1 = esfs(params)
        return -sfs_loglik(afs, e1, sequence_length, theta)

    # Outer vmap: Parallelize across vec_values
    batched_neg_loglik = vmap(evaluate_at_vec)  # in_axes=0 is default

    # 3. Compute all values (runs on GPU/TPU if available)
    results = batched_neg_loglik(vec_values) 
    # results = lax.map(evaluate_at_vec, vec_values)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_likelihood(demo, data_list, cfg_list, paths, vec_values, afs, afs_samples, theta=None, sequence_length=None, recombination_rate=1e-8, t_min=1e-8, num_t=2000, k=2):
    import matplotlib.pyplot as plt

    path_order: List[Var] = list(paths)
    data_pad, cfg_mat, deme_names, max_indices, unique_cfg, matching_indices = process_data(data_list, cfg_list)
    num_samples = len(max_indices)
    first_columns = data_pad[:, :, 0]
    # Compute global max (single float value)
    global_max = jnp.max(first_columns)
    print(global_max)
    t_breaks = jnp.insert(jnp.geomspace(t_min, global_max, num_t), 0, 0.0)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    def compute_loglik(c_map, c_index, data, max_index):
        c = c_map[c_index]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(eta, rho, data, max_index)
    
    def sfs_loglik(afs, esfs, sequence_length, theta):
        afs = afs.flatten()[1:-1]
        esfs = esfs.flatten()[1:-1]
        
        if theta:
            assert(sequence_length)
            tmp = esfs * sequence_length * theta
            return jnp.sum(-tmp + xlogy(afs, tmp))
        else:
            return jnp.sum(xlogy(afs, esfs/esfs.sum()))
    
    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        params = _vec_to_dict_jax(vec_array, path_order)

        c_map = jax.vmap(lambda cfg: iicr_call(params=params, t=t_breaks, num_samples=dict(zip(deme_names, cfg)))["c"])(
            jnp.array(unique_cfg)
        )
        
        # Batched over cfg_mat and all_tmrca_spans 
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0, 0))(c_map, matching_indices, data_pad, max_indices)

        e1 = esfs(params)
        return (-jnp.sum(batched_loglik) / num_samples) + -sfs_loglik(afs, e1, sequence_length, theta) # Same as original neg_loglik

    # Outer vmap: Parallelize across vec_values
    batched_neg_loglik = vmap(evaluate_at_vec)  # in_axes=0 is default

    # 3. Compute all values (runs on GPU/TPU if available)
    results = batched_neg_loglik(vec_values) 
    # results = lax.map(evaluate_at_vec, vec_values)

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS and IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def fit(
    demo,
    data_list, 
    cfg_list,
    paths: Params,
    afs,
    afs_samples,
    *,
    k: int = 2,
    t_min: float = 1e-8,
    # t_max: float,
    num_t: int = 2000,
    method: str = "trust-constr",
    options: Optional[dict] = None,
    recombination_rate: float = 1e-8,
    sequence_length: float = None,
    theta: float = None,
):
    data_pad, cfg_mat, deme_names, max_indices, unique_cfg, matching_indices = process_data(data_list, cfg_list)
    num_samples = len(max_indices)

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
    t_breaks = jnp.insert(jnp.geomspace(t_min, global_max, num_t), 0, 0.0)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    def compute_loglik(c_map, c_index, data, max_index):
        c = c_map[c_index]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(eta, rho, data, max_index)\
    
    def sfs_loglik(afs, esfs, sequence_length, theta):
        afs = afs.flatten()[1:-1]
        esfs = esfs.flatten()[1:-1]
        
        if theta:
            assert(sequence_length)
            tmp = esfs * sequence_length * theta
            return jnp.sum(-tmp + xlogy(afs, tmp))
        else:
            return jnp.sum(xlogy(afs, esfs/esfs.sum()))
    
    @jax.value_and_grad
    def neg_loglik(vec):
        params = _vec_to_dict_jax(vec, path_order)
        c_map = jax.vmap(lambda cfg: iicr_call(params=params, t=t_breaks, num_samples=dict(zip(deme_names, cfg)))["c"])(
            jnp.array(unique_cfg)
        )
        
        # Batched over cfg_mat and all_tmrca_spans 
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0, 0))(c_map, matching_indices, data_pad, max_indices)
        
        likelihood = jnp.sum(batched_loglik)
        e1 = esfs(params)

        return (-likelihood / num_samples) + -sfs_loglik(afs, e1, sequence_length, theta)

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
