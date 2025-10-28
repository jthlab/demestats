from demesinfer.sfs import ExpectedSFS
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import jax.numpy as jnp
from jax import lax, vmap
import jax
import matplotlib.pyplot as plt
from demesinfer.loglik.sfs_loglik import prepare_projection, projection_sfs_loglik, sfs_loglik
from demesinfer.fit.util import _vec_to_dict_jax, process_data, apply_jit
import numpy as np
from demesinfer.iicr import IICRCurve
from phlashlib.loglik import loglik as phlashlib_loglik
from phlashlib.iicr import PiecewiseConstant
from fit_phlashlib import process_base_model, _compute_phlashlib_likelihood
from fit_sfs import _compute_sfs_likelihood

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]

def plot_sfs_likelihood(demo, paths, vec_values, afs, afs_samples, num_projections = 200, seed = 5, projection=False, theta=None, sequence_length=None):
    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    args = (path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs)
    evaluate_at_vec = apply_jit(_compute_sfs_likelihood, *args)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_sfs_contour(demo, paths, param1_vals, param2_vals, afs, afs_samples, num_projections = 200, seed = 5, projection=False, theta=None, sequence_length=None):
    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None
    
    args = (path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs)
    evaluate_at_vec = apply_jit(_compute_sfs_likelihood, *args)

    def compute_for_param1(param1_val):
        def compute_for_param2(param2_val):
            vec_array = jnp.array([param1_val, param2_val])
            evaluate_at_vec(vec_array)
        
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

## The next two functions are for plotting iicr with sequence data using phlashlib, those code need some cleaning ##
def plot_phlashlib_likelihood(demo, het_matrix, cfg_list, paths, vec_values, num_timepoints=33, recombination_rate=1e-8 * 100, theta=1e-8 * 100, k=2):
    import matplotlib.pyplot as plt

    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    num_samples = len(cfg_mat)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(deme_names, unique_cfg, iicr, num_timepoints)
    args = (path_order, times, deme_names, unique_cfg, matching_indices, het_matrix, iicr_call, theta, recombination_rate)

    evaluate_at_vec = apply_jit(_compute_phlashlib_likelihood, *args)
    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_iicr_and_sfs_likelihood(demo, het_matrix, cfg_list, paths, vec_values, afs, afs_samples, num_projections=200, seed=5, projection=False, sequence_length = None, recombination_rate=1e-8 * 100, theta=1e-8 * 100, k=2):
    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    # het_matrix = data
    key = jax.random.PRNGKey(2)
    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    num_samples = len(cfg_mat)
    rho = theta
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    def process_base_model(cfg):
        curve = iicr.curve(num_samples=dict(zip(deme_names, cfg)))
        timepoints = jax.vmap(curve.quantile)(jnp.linspace(0, 1, 64)[1:-1])
        timepoints = jnp.insert(timepoints, 0, 0.0)
        return timepoints

    times = jax.vmap(process_base_model)(jnp.array(unique_cfg))

    def compute_loglik(c_map, c_index, data, times):
        c = c_map[c_index]
        t = times[c_index]
        eta = PiecewiseConstant(c=c, t=t)
        return phlashlib_loglik(data, eta, t, theta, rho)
    
    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        params = _vec_to_dict_jax(vec_array, path_order)

        c_map = jax.vmap(lambda cfg, time: iicr_call(params=params, t=time, num_samples=dict(zip(deme_names, cfg)))["c"])(
            jnp.array(unique_cfg), times
        )
        
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0, None))(c_map, matching_indices, het_matrix, times)
        
        if projection:
            tp = jax.jit(lambda X: esfs.tensor_prod(X, params))
            return -projection_sfs_loglik(tp, proj_dict, einsum_str, input_arrays, sequence_length, theta=None) + -jnp.sum(batched_loglik) / num_samples
        else:
            e1 = esfs(params)
            return -sfs_loglik(afs, e1, sequence_length, theta)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS + IICR Composite Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results