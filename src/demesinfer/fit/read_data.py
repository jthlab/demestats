#### THIS IS A TEMP FILE ####
# Purpose of this file is for reading in data until we pushes an official github 
# for reading data from ts, vcf, plink, etc using bio2zarr : )

import xarray as xr
import dask.array as da
import numpy as np
import sparse
import jax
import jax.random as jr
import sgkit
import bio2zarr.tskit
import tempfile

def zarr2het_specific_comparison(ds: xr.Dataset, samples: list[tuple[str, str]], w: int = 100):
    """
    Compares allele 0 from the first sample of a pair to allele 1 from the
    second sample of the pair, for a list of sample pairs.
    """
    # 1. Prepare coordinates and dimensions
    ds = ds.assign_coords(samples=ds.sample_id)
    pos = ds.variant_position.to_numpy()
    L = int(pos.max() + 1)
    N = len(samples)
    
    # Unpack sample pairs for clear selection
    s0_ids, s1_ids = np.transpose(samples)

    # Pluck first chrom genotype from sid 0, second chrom genotype from sid 1
    gt0 = ds.call_genotype.sel(samples=s0_ids, ploidy=0)
    gt1 = ds.call_genotype.sel(samples=s1_ids, ploidy=1)
    
    het = (gt0.data != gt1.data).T
    
    # Find the indices (pair_idx, variant_idx) where `het` is True.
    # This is a lazy dask operation.
    pair_idx, variant_idx = da.nonzero(het)
    
    # Map the variant indices to their actual genomic positions.
    genomic_pos = pos[variant_idx]
    
    # Now, compute only the coordinates, which is much more memory-efficient
    # than computing the full dense `is_different` matrix.
    final_rows = pair_idx.compute()
    final_cols = genomic_pos
    
    # The data for all these coordinates is simply `True`.
    final_data = np.ones(final_rows.shape[0], dtype=np.uint8)

    # Create the sparse matrix in COO format, which is designed for this.
    coo_array = sparse.COO(
        coords=[final_rows, final_cols],
        data=1,
        shape=(N, L),
    )

    # Convert to CSR for efficiency and wrap in a dask array.
    ret = da.from_array(sparse.GCXS.from_coo(coo_array, compressed_axes=(1,)))

    # Window the data
    return da.coarsen(np.max, ret, {1: w}, trim_excess=True)

def compile(ts, subkey, a=None, b=None):
    # using a set to pull out all unique populations that the samples can possibly belong to
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    if a == None and b == None:
        samples = jax.random.choice(subkey, ts.num_samples/2, shape=(2,), replace=False)
        a, b = samples[0].item(0), samples[1].item(0)

    pop_cfg[ts.population(ts.node(a*2).population).metadata["name"]] += 1
    pop_cfg[ts.population(ts.node(b*2).population).metadata["name"]] += 1

    return pop_cfg, (a, b)

def get_het_data_from_ts(ts, num_samples=100, option="random", seed=2, window_size=100, mask=None):
    key=jr.PRNGKey(seed) 
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

    d = tempfile.TemporaryDirectory() 
    bio2zarr.tskit.convert(ts, d.name + "/ts")
    ds = sgkit.load_dataset(d.name + "/ts")

    names = ds["sample_id"].to_numpy()
    result = zarr2het_specific_comparison(ds, names[all_config], 100).compute().todense()
    
    return result, cfg_list



