from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import msprime as msp
from scipy.optimize import LinearConstraint, minimize
import jax.random as jr
from jax import vmap, lax 

from demesinfer.coal_rate import PiecewiseConstant
from demesinfer.constr import EventTree, constraints_for
from demesinfer.iicr import IICRCurve
from phlashlib.loglik import loglik
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

from intervaltree import IntervalTree
import tqdm.auto as tqdm
import tskit

def _read_ts(
    ts: tskit.TreeSequence,
    nodes: list[tuple[int, int]],
    window_size: int,
    progress: bool = False,
) -> np.ndarray:
    nodes_flat = list({x for t in nodes for x in t})
    node_inds = np.array([[nodes_flat.index(x) for x in t] for t in nodes])
    N = len(nodes)
    L = int(np.ceil(ts.get_sequence_length() / window_size))
    G = np.zeros([N, L], dtype=np.int8)
    with tqdm.tqdm(
        ts.variants(samples=nodes_flat, copy=False),
        total=ts.num_sites,
        disable=not progress,
    ) as pbar:
        pbar.set_description("Reading tree sequence")
        for v in pbar:
            g = v.genotypes[node_inds]
            ell = int(v.position / window_size)
            G[:, ell] += g[:, 0] != g[:, 1]
    return G

def get_data(ts, nodes=None, window_size=100, mask=None):
    # form interval tree for masking
    L = int(ts.get_sequence_length())
    mask = mask or []
    
    if nodes is None:
        nodes = [tuple(i.nodes) for i in ts.individuals()]

    tr = IntervalTree.from_tuples([(0, L)])
    for a, b in mask:
        tr.chop(a, b)
    # compute breakpoints
    bp = np.array([x for i in tr for x in [i.begin, i.end]])
    assert len(set(bp)) == len(bp)
    assert (bp == np.sort(bp)).all()
    if bp[0] != 0.0:
        bp = np.insert(bp, 0, 0.0)
    if bp[-1] != L:
        bp = np.append(bp, L)
    mid = (bp[:-1] + bp[1:]) / 2.0
    unmasked = [bool(tr[m]) for m in mid]
    nodes_flat = list({x for t in nodes for x in t})
    afs = ts.allele_frequency_spectrum(
        sample_sets=[nodes_flat], windows=bp, polarised=True, span_normalise=False
    )[unmasked].sum(0)[1:-1]
    het_matrix = _read_ts(ts, nodes, window_size)
    # now mask out columns of the het matrix based on interval
    # overlap
    tr = IntervalTree.from_tuples(mask)
    column_mask = [
        bool(tr[a : a + window_size]) for a in range(0, L, window_size)
    ]
    assert len(column_mask) == het_matrix.shape[1]
    # set mask out these columns
    het_matrix[:, column_mask] = -1
    return dict(afs=afs, het_matrix=het_matrix)

def compile(ts, subkey, a=None, b=None):
    # using a set to pull out all unique populations that the samples can possibly belong to
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    if a == None and b == None:
        samples = jax.random.choice(subkey, ts.num_samples, shape=(2,), replace=False)
        a, b = samples[0].item(0), samples[1].item(0)

    pop_cfg[ts.population(ts.node(a).population).metadata["name"]] += 1
    pop_cfg[ts.population(ts.node(b).population).metadata["name"]] += 1
    return pop_cfg, (a, b)

def get_het_data(ts, key=jax.random.PRNGKey(2), num_samples=200, option="random", window_size=100, mask=None):
    cfg_list = []
    all_config=[]
    key, subkey = jr.split(key)
    if option == "random":
        for i in range(num_samples):
            cfg, pair = compile(ts, subkey)
            cfg_list.append(cfg)
            all_config.append(pair)
            key, subkey = jr.split(key)
    elif option == "all":
        from itertools import combinations
        all_config = list(combinations(ts.samples(), 2))
        for a, b in all_config:
            cfg = compile(ts, subkey, a, b)
            cfg_list.append(cfg)
    elif option == "unphased":
        all_config = ts.samples().reshape(-1, 2)
        for a, b in all_config:
            cfg = compile(ts, subkey, a, b)
            cfg_list.append(cfg)

    result = get_data(ts, all_config, window_size, mask)
    return result, cfg_list

def process_data(cfg_list):
    num_samples = len(cfg_list)

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
    
    return cfg_mat, deme_names, unique_cfg, matching_indices

def plot_iicr_likelihood(demo, data, cfg_list, paths, vec_values, recombination_rate=1e-8, theta=1e-8, t_min=1e-8, t_max=1e7, num_t=2000, k=2):
    import matplotlib.pyplot as plt

    het_matrix = data["het_matrix"]
    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    num_samples = len(cfg_mat)
    t_breaks = jnp.insert(jnp.geomspace(t_min, t_max, num_t), 0, 0.0)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)

    def compute_loglik(c_map, c_index, data, max_index):
        c = c_map[c_index]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(data, eta, t_breaks, theta, rho)
    
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
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0))(c_map, matching_indices, het_matrix)
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

def plot_likelihood(demo, data, cfg_list, paths, vec_values, afs_samples, theta=None, sequence_length=None, recombination_rate=1e-8, t_min=1e-8, t_max=1e7, num_t=2000, k=2):
    import matplotlib.pyplot as plt

    path_order: List[Var] = list(paths)
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    het_matrix = data["het_matrix"]
    afs = data["afs"] # double check that this really is the AFS, especially check if it's already removed the first and last entry and whether I need to even flatten it
    num_samples = len(matching_indices)
    t_breaks = jnp.insert(jnp.geomspace(t_min, t_max, num_t), 0, 0.0)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    def compute_loglik(c_map, c_index, data, max_index):
        c = c_map[c_index]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(data, eta, t_breaks, theta, rho)
    
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
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0))(c_map, matching_indices, het_matrix)

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
    data, 
    cfg_list,
    paths: Params,
    afs,
    afs_samples,
    *,
    k: int = 2,
    t_min: float = 1e-8,
    t_max: float = 1e7,
    # t_max: float,
    num_t: int = 2000,
    method: str = "trust-constr",
    options: Optional[dict] = None,
    recombination_rate: float = 1e-8,
    sequence_length: float = None,
    theta: float = None,
):
    cfg_mat, deme_names, unique_cfg, matching_indices = process_data(cfg_list)
    het_matrix = data["het_matrix"]
    afs = data["afs"]
    num_samples = len(matching_indices)

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

    t_breaks = jnp.insert(jnp.geomspace(t_min, t_max, num_t), 0, 0.0)
    rho = recombination_rate
    iicr = IICRCurve(demo=demo, k=k)
    iicr_call = jax.jit(iicr.__call__)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    def compute_loglik(c_map, c_index, data, max_index):
        c = c_map[c_index]
        eta = PiecewiseConstant(c=c, t=t_breaks)
        return loglik(data, eta, t_breaks, theta, rho)
    
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
        batched_loglik = vmap(compute_loglik, in_axes=(None, 0, 0))(c_map, matching_indices, het_matrix)
        
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
