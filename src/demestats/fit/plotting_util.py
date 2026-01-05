from typing import Any, List, Mapping, Set, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import lax

from demestats.fit.fit_arg import _compute_arg_likelihood
from demestats.fit.fit_composite_arg_sfs import _compute_composite_arg_sfs_likelihood
from demestats.fit.fit_composite_phlashlib_sfs import (
    _compute_composite_phlashlib_sfs_likelihood,
)
from demestats.fit.fit_phlashlib import (
    _compute_phlashlib_likelihood,
    process_base_model,
)
from demestats.fit.fit_sfs import _compute_sfs_likelihood
from demestats.fit.util import apply_jit, process_arg_data, process_data, reformat_data
from demestats.iicr import IICRCurve
from demestats.loglik.sfs_loglik import prepare_projection
from demestats.sfs import ExpectedSFS

Path = Tuple[Any, ...]
Var = Path | Set[Path]
Params = Mapping[Var, float]


def plot_sfs_likelihood(
    demo,
    paths,
    vec_values,
    afs,
    afs_samples,
    num_projections=200,
    seed=5,
    projection=False,
    theta=None,
    sequence_length=None,
):
    """
    Plot the negative log-likelihood landscape for SFS parameters.

    Parameters
    ----------
    demo : demes.Graph
        a ``demes`` graph
    paths : dictionary
        Dictionary of a single variable path to evaluate.
    vec_values : jax.Array, numpy.ndarray
        Array of parameter values to evaluate.
    afs : array_like
        Observed allele frequency spectrum.
    afs_samples : dictionary
        Dictionary specifying the number of haploids in each population for the afs
    num_projections : int, optional
        Number of random projections to use (default: 200).
    seed : int, optional
        Random seed for projection (default: 5).
    projection : bool, optional
        Whether to use random projections (default: False).
    theta : float, optional
        Population-scaled mutation rate. Required for Poisson likelihood.
    sequence_length : int, optional
        Sequence length. Required if ``theta`` is provided.

    Returns
    -------
    jax.Array
        Array of negative log-likelihood values for each parameter value.

    Notes
    -----
    This function computes and plots the negative log-likelihood landscape
    for a given parameter across a range of values. When ``projection=True``,
    random projections are used to reduce dimensionality and accelerate
    computation.

    Example
    -------
    ::
        paths = {
            frozenset({
                ("demes", 0, "epochs", 0, "end_size"),
                ("demes", 0, "epochs", 0, "start_size"),
            }): 4000.,
        }
        vec_values = jnp.linspace(4000, 6000, 50)
        result = plot_sfs_likelihood(demo.to_demes(), paths, vec_values, afs, afs_samples)

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    """

    path_order: List[Var] = list(paths)
    esfs_obj = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    args_nonstatic = (
        path_order,
        proj_dict,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
    )
    args_static = (esfs_obj, einsum_str)
    evaluate_at_vec = apply_jit(_compute_sfs_likelihood, args_nonstatic, args_static)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, "r-", linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results


def plot_sfs_contour(
    demo,
    paths,
    param1_vals,
    param2_vals,
    afs,
    afs_samples,
    num_projections=200,
    seed=5,
    projection=False,
    theta=None,
    sequence_length=None,
):
    """
    Plot 2D contour plot of negative log-likelihood for two parameters.

    Parameters
    ----------
    demo : demes.Graph
        a ``demes`` graph
    paths : dictionary
        Dictionary of a single variable path to evaluate.
    param1_vals : jax.Array, numpy.ndarray
        Array of values for the first parameter.
    param2_vals : jax.Array, numpy.ndarray
        Array of values for the second parameter.
    afs : array_like
        Observed allele frequency spectrum.
    afs_samples : dictionary
        Dictionary specifying the number of haploids in each population for the afs
    num_projections : int, optional
        Number of random projections to use (default: 200).
    seed : int, optional
        Random seed for projection (default: 5).
    projection : bool, optional
        Whether to use random projections (default: False).
    theta : float, optional
        Population-scaled mutation rate. Required for Poisson likelihood.
    sequence_length : int, optional
        Sequence length. Required if ``theta`` is provided.

    Returns
    -------
    tuple: (param1_grid, param2_grid, log_likelihood_grid) containing the
    meshgrid arrays and computed negative log-likelihood values.

    Notes
    -----
    This function computes and visualizes a 2D likelihood landscape for
    two parameters. The contour plot shows regions of high and low likelihood.
    When ``projection=True``, random projections are used to reduce dimensionality
    and accelerate computation.

    Example
    -------
    ::
        paths = {
            frozenset({
                ("demes", 1, "epochs", 0, "end_size"),
                ("demes", 1, "epochs", 0, "start_size"),
            }): 4000.,
            frozenset({
                ("demes", 2, "epochs", 0, "end_size"),
                ("demes", 2, "epochs", 0, "start_size"),
            }): 4000.,
        }

        param1_vals = jnp.linspace(4000, 6000, 10)
        param2_vals = jnp.linspace(4000, 6000, 10)

        result = plot_sfs_contour(demo.to_demes(), paths, param1_vals, param2_vals, afs, afs_samples)

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    """
    path_order: List[Var] = list(paths)
    esfs_obj = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    args_nonstatic = (
        path_order,
        proj_dict,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
    )
    args_static = (esfs_obj, einsum_str)
    evaluate_at_vec = apply_jit(_compute_sfs_likelihood, args_nonstatic, args_static)

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
    contour = plt.contourf(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Negative Log-Likelihood")

    contour_lines = plt.contour(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        colors="black",
        linewidths=0.5,
        alpha=0.5,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title("Negative Log-Likelihood Contour Plot")
    plt.show()

    return param1_grid_np, param2_grid_np, log_likelihood_grid_np


## The next two functions are for plotting iicr using phlashlib, those code need some cleaning ##
def plot_phlashlib_likelihood(
    demo,
    het_matrix,
    cfg_list,
    paths,
    vec_values,
    num_timepoints=33,
    recombination_rate=1e-8 * 100,
    theta=1e-8 * 100,
    k=2,
):
    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(
        deme_names, unique_cfg, iicr, num_timepoints
    )
    args = (
        path_order,
        times,
        deme_names,
        unique_cfg,
        matching_indices,
        het_matrix,
        iicr_call,
        theta,
        recombination_rate,
    )

    evaluate_at_vec = apply_jit(_compute_phlashlib_likelihood, *args)
    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, "r-", linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results


def plot_phlashlib_contour(
    demo,
    het_matrix,
    cfg_list,
    paths,
    param1_vals,
    param2_vals,
    num_timepoints=33,
    recombination_rate=1e-8,
    theta=1e-8,
    window_size=100,
    k=2,
):
    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)
    recombination_rate = recombination_rate * window_size
    theta = theta * window_size

    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(
        deme_names, unique_cfg, iicr, num_timepoints
    )
    args = (
        path_order,
        times,
        deme_names,
        unique_cfg,
        matching_indices,
        het_matrix,
        iicr_call,
        theta,
        recombination_rate,
    )

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
    contour = plt.contourf(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Negative Log-Likelihood")

    contour_lines = plt.contour(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        colors="black",
        linewidths=0.5,
        alpha=0.5,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title("Negative Log-Likelihood Contour Plot")
    plt.show()

    return param1_grid_np, param2_grid_np, log_likelihood_grid_np


## The next two functions are for plotting iicr using arg likelihood, those code need some cleaning ##
def plot_arg_likelihood(
    demo, data_list, cfg_list, paths, vec_values, recombination_rate=1e-8, k=2
):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(
        data_list, cfg_list
    )
    chunking_length = data_pad.shape[1]
    (
        unique_times,
        rearranged_data,
        new_matching_indices,
        new_max_indices,
        associated_indices,
        unique_groups,
        batch_size,
        group_membership,
    ) = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = iicr.curve

    args = (
        path_order,
        unique_cfg,
        deme_names,
        iicr_call,
        recombination_rate,
        unique_times,
        group_membership,
        associated_indices,
        batch_size,
        chunking_length,
        rearranged_data,
        new_max_indices,
    )
    evaluate_at_vec = apply_jit(_compute_arg_likelihood, *args)
    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, "r-", linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results


def plot_arg_contour(
    demo, data_list, cfg_list, paths, param1_vals, param2_vals, recombination_rate
):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(
        data_list, cfg_list
    )
    chunking_length = data_pad.shape[1]
    (
        unique_times,
        rearranged_data,
        new_matching_indices,
        new_max_indices,
        associated_indices,
        unique_groups,
        batch_size,
        group_membership,
    ) = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    recombination_rate = 1e-8
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = iicr.curve

    args = (
        path_order,
        unique_cfg,
        deme_names,
        iicr_call,
        recombination_rate,
        unique_times,
        group_membership,
        associated_indices,
        batch_size,
        chunking_length,
        rearranged_data,
        new_max_indices,
    )
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
    contour = plt.contourf(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Negative Log-Likelihood")

    contour_lines = plt.contour(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        colors="black",
        linewidths=0.5,
        alpha=0.5,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title("Negative Log-Likelihood Contour Plot")
    plt.show()

    return param1_grid_np, param2_grid_np, log_likelihood_grid_np


## The next two functions are for plotting sfs and arg likelihood, those code need some cleaning ##
def plot_arg_sfs_likelihood(
    demo,
    data_list,
    cfg_list,
    afs,
    afs_samples,
    paths,
    vec_values,
    recombination_rate=1e-8,
    k=2,
    sequence_length=None,
    theta=None,
    num_projections=200,
    seed=5,
    projection=False,
):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(
        data_list, cfg_list
    )
    chunking_length = data_pad.shape[1]
    (
        unique_times,
        rearranged_data,
        new_matching_indices,
        new_max_indices,
        associated_indices,
        unique_groups,
        batch_size,
        group_membership,
    ) = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = iicr.curve

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    first_element = jnp.atleast_1d(vec_values[0])
    n1 = _compute_arg_likelihood(
        first_element,
        path_order,
        unique_cfg,
        deme_names,
        iicr_call,
        recombination_rate,
        unique_times,
        group_membership,
        associated_indices,
        batch_size,
        chunking_length,
        rearranged_data,
        new_max_indices,
    )
    n2 = _compute_sfs_likelihood(
        first_element,
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
    )

    args = (
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
        unique_cfg,
        deme_names,
        iicr_call,
        recombination_rate,
        unique_times,
        group_membership,
        associated_indices,
        batch_size,
        chunking_length,
        rearranged_data,
        new_max_indices,
        n1,
        n2,
    )
    evaluate_at_vec = apply_jit(_compute_composite_arg_sfs_likelihood, *args)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, "r-", linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results


def plot_arg_sfs_contour(
    demo,
    data_list,
    cfg_list,
    afs,
    afs_samples,
    paths,
    param1_vals,
    param2_vals,
    recombination_rate=1e-8,
    k=2,
    sequence_length=None,
    theta=None,
    num_projections=200,
    seed=5,
    projection=False,
):
    path_order: List[Var] = list(paths)
    data_pad, deme_names, max_indices, unique_cfg, matching_indices = process_arg_data(
        data_list, cfg_list
    )
    chunking_length = data_pad.shape[1]
    (
        unique_times,
        rearranged_data,
        new_matching_indices,
        new_max_indices,
        associated_indices,
        unique_groups,
        batch_size,
        group_membership,
    ) = reformat_data(data_pad, matching_indices, max_indices, chunking_length)

    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = iicr.curve

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    first_element = jnp.array([param1_vals[0], param2_vals[0]])
    n1 = _compute_arg_likelihood(
        first_element,
        path_order,
        unique_cfg,
        deme_names,
        iicr_call,
        recombination_rate,
        unique_times,
        group_membership,
        associated_indices,
        batch_size,
        chunking_length,
        rearranged_data,
        new_max_indices,
    )
    n2 = _compute_sfs_likelihood(
        first_element,
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
    )

    args = (
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        theta,
        projection,
        afs,
        unique_cfg,
        deme_names,
        iicr_call,
        recombination_rate,
        unique_times,
        group_membership,
        associated_indices,
        batch_size,
        chunking_length,
        rearranged_data,
        new_max_indices,
        n1,
        n2,
    )
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
    contour = plt.contourf(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Negative Log-Likelihood")

    contour_lines = plt.contour(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        colors="black",
        linewidths=0.5,
        alpha=0.5,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title("Negative Log-Likelihood Contour Plot")
    plt.show()

    return param1_grid_np, param2_grid_np, log_likelihood_grid_np


## The next two functions are for plotting sfs and arg likelihood, those code need some cleaning ##
def plot_phlashlib_sfs_likelihood(
    demo,
    het_matrix,
    cfg_list,
    afs,
    afs_samples,
    paths,
    vec_values,
    num_timepoints=33,
    recombination_rate=1e-8,
    theta=1e-8,
    k=2,
    window_size=100,
    sequence_length=None,
    mutation_rate_sfs=None,
    num_projections=200,
    seed=5,
    projection=False,
):
    path_order: List[Var] = list(paths)
    het_matrix = jnp.array(het_matrix)

    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    unique_cfg = jnp.array(unique_cfg)

    rho = recombination_rate * window_size
    theta = theta * window_size
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = jax.jit(iicr.__call__)

    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(
        deme_names, unique_cfg, iicr, num_timepoints
    )

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    first_element = jnp.atleast_1d(vec_values[0])
    n1 = _compute_phlashlib_likelihood(
        first_element,
        path_order,
        times,
        deme_names,
        unique_cfg,
        matching_indices,
        het_matrix,
        iicr_call,
        theta,
        rho,
    )
    n2 = _compute_sfs_likelihood(
        first_element,
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        mutation_rate_sfs,
        projection,
        afs,
    )

    args = (
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        mutation_rate_sfs,
        projection,
        afs,
        times,
        deme_names,
        unique_cfg,
        matching_indices,
        het_matrix,
        iicr_call,
        theta,
        rho,
        n1,
        n2,
    )
    evaluate_at_vec = apply_jit(_compute_composite_phlashlib_sfs_likelihood, *args)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, "r-", linewidth=2)
    plt.xlabel("vec value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("IICR Likelihood Landscape")
    plt.grid(True)
    plt.show()

    return results


def plot_phlashlib_sfs_contour(
    demo,
    het_matrix,
    cfg_list,
    afs,
    afs_samples,
    paths,
    param1_vals,
    param2_vals,
    num_timepoints=33,
    recombination_rate=1e-8,
    theta=1e-8,
    k=2,
    window_size=100,
    sequence_length=None,
    mutation_rate_sfs=None,
    num_projections=200,
    seed=5,
    projection=False,
):
    path_order: List[Var] = list(paths)
    het_matrix = jnp.array(het_matrix)

    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    unique_cfg = jnp.array(unique_cfg)

    rho = recombination_rate * window_size
    theta = theta * window_size
    iicr = IICRCurve(demo=demo, k=2)
    iicr_call = jax.jit(iicr.__call__)

    times = jax.vmap(process_base_model, in_axes=(None, 0, None, None))(
        deme_names, unique_cfg, iicr, num_timepoints
    )

    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    if projection:
        proj_dict, einsum_str, input_arrays = prepare_projection(
            afs, afs_samples, sequence_length, num_projections, seed
        )
    else:
        proj_dict, einsum_str, input_arrays = None, None, None

    first_element = jnp.array([param1_vals[0], param2_vals[0]])
    n1 = _compute_phlashlib_likelihood(
        first_element,
        path_order,
        times,
        deme_names,
        unique_cfg,
        matching_indices,
        het_matrix,
        iicr_call,
        theta,
        rho,
    )
    n2 = _compute_sfs_likelihood(
        first_element,
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        mutation_rate_sfs,
        projection,
        afs,
    )

    args = (
        path_order,
        esfs,
        proj_dict,
        einsum_str,
        input_arrays,
        sequence_length,
        mutation_rate_sfs,
        projection,
        afs,
        times,
        deme_names,
        unique_cfg,
        matching_indices,
        het_matrix,
        iicr_call,
        theta,
        rho,
        n1,
        n2,
    )
    evaluate_at_vec = apply_jit(_compute_composite_phlashlib_sfs_likelihood, *args)

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
    contour = plt.contourf(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        cmap="viridis",
    )
    plt.colorbar(contour, label="Negative Log-Likelihood")

    contour_lines = plt.contour(
        param1_grid_np,
        param2_grid_np,
        log_likelihood_grid_np.T,
        levels=20,
        colors="black",
        linewidths=0.5,
        alpha=0.5,
    )
    plt.clabel(contour_lines, inline=True, fontsize=8)

    plt.xlabel("Parameter 1")
    plt.ylabel("Parameter 2")
    plt.title("Negative Log-Likelihood Contour Plot")
    plt.show()

    return param1_grid_np, param2_grid_np, log_likelihood_grid_np
