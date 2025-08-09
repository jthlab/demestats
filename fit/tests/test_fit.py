import jax
import jax.numpy as jnp
from fit.fit import sample_tmrca_spans
import numpy as np
import msprime as msp

def create_ts():
    import msprime as msp
    import demes
    import demesdraw

    demo = msp.Demography()
    demo.add_population(initial_size = 10000, name = "anc")
    demo.add_population(initial_size = 1e4, name = "P0")
    demo.add_population(initial_size = 1e4, name = "P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.001)
    tmp = [f"P{i}" for i in range(2)]
    demo.add_population_split(time = 1000, derived=tmp, ancestral="anc")
    g = demo.to_demes()
    sample_size = 10
    samples = {f"P{i}": sample_size for i in range(2)}
    anc = msp.sim_ancestry(samples=samples, demography=demo, recombination_rate=1e-8, sequence_length=1e7, random_seed = 12)
    ts = msp.sim_mutations(anc, rate=1e-8, random_seed = 12)
    return ts, g

def sample_tmrca_spans(ts, subkey=jax.random.PRNGKey(1), num_pop=2):
    samples = jax.random.choice(subkey, ts.num_samples, shape=(2,), replace=False)
    sample1, sample2 = samples[0], samples[1]

    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}
    pop_cfg[ts.population(ts.node(sample1.item(0)).population).metadata["name"]] += 1
    pop_cfg[ts.population(ts.node(sample2.item(0)).population).metadata["name"]] += 1

    # Precompute all TMRCAs and spans into arrays
    tmrcas = []
    spans = []
    for tree in ts.trees():
        spans.append(tree.interval.right - tree.interval.left)
        tmrcas.append(tree.tmrca(sample1, sample2))
    
    # Convert to JAX arrays
    tmrcas = jnp.array(tmrcas)  # Shape: (num_trees,)
    spans = jnp.array(spans)    # Shape: (num_trees,)
    tmrcas_spans = jnp.stack([tmrcas, spans], axis=1)  # Shape: (num_trees, 2)

    # Merge consecutive spans with same TMRCA
    def merge_spans(carry, x):
        current_tmrca, current_span, idx, output = carry
        tmrca, span = x
        
        # Update each component individually
        new_tmrca = jnp.where(tmrca == current_tmrca, current_tmrca, tmrca)
        new_span = jnp.where(tmrca == current_tmrca, current_span + span, span)
        new_idx = jnp.where(tmrca == current_tmrca, idx, idx + 1)
        new_output = jnp.where(
            tmrca == current_tmrca, 
            output, 
            output.at[idx].set(jnp.array([current_tmrca, current_span]))
        )
        
        return (new_tmrca, new_span, new_idx, new_output), None

    init_carry = (tmrcas_spans[0, 0], 0.0, 0, jnp.full((ts.num_trees, 2), jnp.array([1.0, 0.0])))
    final_carry, _ = jax.lax.scan(merge_spans, init_carry, tmrcas_spans)
    final_tmrca, final_span, _, final_output = final_carry
    final_output = final_output.at[-1].set(jnp.array([final_tmrca, final_span]))
    is_ones = jnp.all(final_output == jnp.array([1.0, 0.0]), axis=1)
    reordered_arr = jnp.concatenate([final_output[~is_ones], final_output[is_ones]])

    return reordered_arr, pop_cfg, final_output[~is_ones]

def compile(ts, subkey):
    # using a set to pull out all unique populations that the samples can possibly belong to
    pop_cfg = {ts.population(ts.node(n).population).metadata["name"] for n in ts.samples()}
    pop_cfg = {pop_name: 0 for pop_name in pop_cfg}

    samples = jax.random.choice(subkey, ts.num_samples, shape=(2,), replace=False)
    a, b = samples[0].item(0), samples[1].item(0)
    tmp = pop_cfg.copy()
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
    tmp[ts.population(ts.node(a).population).metadata["name"]] += 1
    tmp[ts.population(ts.node(b).population).metadata["name"]] += 1
    return data, tmp

def test_sample_tmrca_spans():
    key = jax.random.PRNGKey(0)
    ts, _ = create_ts()
    all_tmrca_spans_truth = []
    cfg_list_truth = []
    for i in range(5):
        key, subkey = jax.random.split(key)
        data, tmp = compile(ts, subkey)
        all_tmrca_spans_truth.append(data)
        cfg_list_truth.append(tmp)

    all_tmrca_spans = []
    cfg_list = []
    all_tmrca_spans_no_pad = []
    for i in range(5):
        key, subkey = jax.random.split(key)
        tmrca_spans, pop_cfg, tmrca_spans_no_pad = sample_tmrca_spans(ts, subkey)
        all_tmrca_spans.append(tmrca_spans)
        cfg_list.append(pop_cfg)
        all_tmrca_spans_no_pad.append(tmrca_spans_no_pad)

    for i in range(5):
        np.testing.assert_array_equal(all_tmrca_spans_truth[i], all_tmrca_spans_no_pad[i], err_msg="Arrays are not equal")

    assert(cfg_list_truth == cfg_list)
