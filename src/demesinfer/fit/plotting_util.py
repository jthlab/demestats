from demesinfer.sfs import ExpectedSFS
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
import jax.numpy as jnp
from jax import lax, vmap
import jax
import matplotlib.pyplot as plt
from demesinfer.loglik.sfs_loglik import prepare_projection, projection_sfs_loglik, sfs_loglik
from demesinfer.fit.util import _vec_to_dict_jax, process_data, apply_jit, process_arg_data, reformat_data
import numpy as np
from demesinfer.iicr import IICRCurve
from phlashlib.loglik import loglik as phlashlib_loglik
from phlashlib.iicr import PiecewiseConstant
from demesinfer.fit.fit_phlashlib import process_base_model, _compute_phlashlib_likelihood
from demesinfer.fit.fit_sfs import _compute_sfs_likelihood
from demesinfer.fit.fit_arg import _compute_arg_likelihood
from demesinfer.fit.fit_composite_arg_sfs import _compute_composite_arg_sfs_likelihood

import matplotlib.pyplot as plt

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

## The next two functions are for plotting iicr using phlashlib, those code need some cleaning ##
def plot_phlashlib_likelihood(demo, het_matrix, cfg_list, paths, vec_values, num_timepoints=33, recombination_rate=1e-8 * 100, theta=1e-8 * 100, k=2):
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

def plot_phlashlib_contour(demo, het_matrix, cfg_list, paths, param1_vals, param2_vals, num_timepoints=33, recombination_rate=1e-8 * 100, theta=1e-8 * 100, k=2):
    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    num_samples = len(cfg_mat)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(deme_names, unique_cfg, iicr, num_timepoints)
    args = (path_order, times, deme_names, unique_cfg, matching_indices, het_matrix, iicr_call, theta, recombination_rate)

    evaluate_at_vec = apply_jit(_compute_phlashlib_likelihood, *args)

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

## The next two functions are for plotting iicr using arg likelihood, those code need some cleaning ##
def plot_arg_likelihood(demo, data_list, cfg_list, paths, vec_values, recombination_rate=1e-8, k=2):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(data_list, cfg_list)
    chunking_length = data_pad.shape[1]
    unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = iicr.curve
        
    args = (path_order, unique_cfg, deme_names, iicr_call, recombination_rate, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices)
    evaluate_at_vec = apply_jit(_compute_arg_likelihood, *args)
    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_arg_contour(demo, data_list, cfg_list, paths, param1_vals, param2_vals, recombination_rate):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(data_list, cfg_list)
    chunking_length = data_pad.shape[1]
    unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    recombination_rate = 1e-8
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = iicr.curve
        
    args = (path_order, unique_cfg, deme_names, iicr_call, recombination_rate, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices)
    evaluate_at_vec = apply_jit(_compute_arg_likelihood, *args)

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

## The next two functions are for plotting sfs and arg likelihood, those code need some cleaning ##
def plot_arg_sfs_likelihood(demo, data_list, cfg_list, afs, afs_samples, paths, vec_values, recombination_rate=1e-8, k=2, sequence_length=None, theta=None, num_projections=200, seed=5, projection=False):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(data_list, cfg_list)
    chunking_length = data_pad.shape[1]
    unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = iicr.curve

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    args = (path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs, unique_cfg, deme_names, iicr_call, recombination_rate, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices, n1, n2)
    evaluate_at_vec = apply_jit(_compute_composite_arg_sfs_likelihood, *args)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, 'r-', linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results

def plot_arg_sfs_contour(demo, data_list, cfg_list, afs, afs_samples, paths, param1_vals, param2_vals, recombination_rate=1e-8, k=2, sequence_length=None, theta=None, num_projections=200, seed=5, projection=False): 
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(data_list, cfg_list)
    chunking_length = data_pad.shape[1]
    unique_times, rearranged_data, new_matching_indices, new_max_indices, associated_indices, unique_groups, batch_size, group_membership = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = iicr.curve

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(afs, afs_samples, sequence_length, num_projections, seed)
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    args = (path_order, esfs, proj_dict, einsum_str, input_arrays, sequence_length, theta, projection, afs, unique_cfg, deme_names, iicr_call, recombination_rate, unique_times, group_membership, associated_indices, batch_size, chunking_length, rearranged_data, new_max_indices, n1, n2)
    evaluate_at_vec = apply_jit(_compute_composite_arg_sfs_likelihood, *args)

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
    

