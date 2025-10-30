import jax.numpy as jnp
import jax
from itertools import combinations
import jax.random as jr

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from jax import lax
import matplotlib.pyplot as plt
from demesinfer.fit.util import apply_jit
import numpy as np
from demesinfer.iicr import IICRCurve
from demesinfer.fit.fit_mrpast import _compute_mrpast_likelihood

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def compile_samples(ts, subkey, tree_list, k, samples=None):
   # using a set to pull out all unique populations that the samples can possibly belong to
   pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
   pop_cfg = {pop_name: 0 for pop_name in pop_cfg}
   tmrca_spans = []

   if samples == None:
       samples = jax.random.choice(subkey, ts.num_samples, shape=(k,), replace=False)
   
   for tree in tree_list:
       configurations = []
       for s1, s2 in combinations(samples, 2):
           configurations.append(tree.tmrca(s1, s2))
       tmrca_spans.append(jnp.min(jnp.array(configurations)))

   for i in samples:
       pop_cfg[ts.population(ts.node(i.item(0)).population).metadata["name"]] += 1

   return jnp.sort(jnp.array(tmrca_spans, dtype=jnp.float64)).flatten(), pop_cfg

def get_tree_from_positions_data(ts, num_samples=50, gap=125000, k=2, seed=42, option="random"):
    key = jax.random.PRNGKey(seed)
    num_trees = jnp.ceil(ts.sequence_length / gap)
    start_position = jax.random.randint(key, (1,), 1, gap+1)
    additive = jnp.arange(num_trees) * gap
    position_index = start_position + additive
    
    tree_list = []
    for pos in position_index:
       tree = ts.at(pos)
       tree_list.append(tree)
    
    data_list = []
    cfg_list = []
    key, subkey = jr.split(key)

    if option=="random":
        for i in range(num_samples):
           data, cfg = compile_samples(ts, key, tree_list, k)
           data_list.append(jnp.array(data))
           cfg_list.append(cfg)
           key, subkey = jr.split(key)

    if option=="all":
        all_configs = list(combinations(jnp.arange(ts.num_samples), k))
        for config in all_configs:
           data, cfg = compile_samples(ts, key, tree_list, k, samples=config)
           data_list.append(jnp.array(data))
           cfg_list.append(cfg)
           key, subkey = jr.split(key)
    
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

def plot_sfs_likelihood(demo, paths, vec_values, data, cfg_list, k=2):
    path_order: List[Var] = list(paths)
    data = jnp.array(data)

    cfg_mat, deme_names = process_data(cfg_list)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    args = (path_order, data, cfg_mat, iicr_call, deme_names)
    evaluate_at_vec = apply_jit(_compute_mrpast_likelihood, *args)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_sfs_contour(demo, paths, param1_vals, param2_vals, data, cfg_list, k=2):
    path_order: List[Var] = list(paths)
    data = jnp.array(data)

    cfg_mat, deme_names = process_data(cfg_list)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    args = (path_order, data, cfg_mat, iicr_call, deme_names)
    evaluate_at_vec = apply_jit(_compute_mrpast_likelihood, *args)

    def compute_for_param1(param1_val):
        def compute_for_param2(param2_val):
            vec_array = jnp.array([param1_val, param2_val])
            return evaluate_at_vec(vec_array)
        
        # Map over param2 values for a fixed param1
        return jax.lax.map(compute_for_param2, param2_vals)
    
    # Map over param1 values
    log_likelihood_grid = jax.lax.map(compute_for_param1, param1_vals)
    
    param1_grid, param2_grid = jnp.meshgrid(param1_vals, param2_vals)
    param1_grid_np = np.array(param1_grid)
    param2_grid_np = np.array(param2_grid)
    log_likelihood_grid_np = np.array(log_likelihood_grid)
    
    plt.figure(figsize=(10, 8))
    
    # Use contourf for filled contours (heatmap instead of just lines)
    contour = plt.contourf(param1_grid_np, param2_grid_np, log_likelihood_grid_np.T, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Negative Log-Likelihood')
    
    contour_lines = plt.contour(param1_grid_np, param2_grid_np, log_likelihood_grid_np.T, levels=20, colors='black', linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8)
    
    plt.xlabel('Parameter 1')
    plt.ylabel('Parameter 2')
    plt.title('Negative Log-Likelihood Contour Plot')
    plt.show()
    
    return param1_grid_np, param2_grid_np, log_likelihood_grid_np