---
jupyter:
  jupytext:
    default_lexer: python
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# CCR: Cross-Coalescent Rate

This note documents how to use the CCR functions and the difference between the
exact colored-CTMC implementation and the mean-field approximation.

## Overview

The CCR measures the instantaneous hazard of the first red-blue coalescence.
The API mirrors `demestats.iicr`:

- `demestats.ccr.CCRCurve`: exact colored lineage-count CTMC (accurate but
  scales poorly with sample size and number of demes).
- `demestats.ccr.CCRMeanFieldCurve`: mean-field ODE approximation (scales well,
  slightly less accurate).

Both return a dict with:

- `c`: the CCR curve evaluated at the requested times.
- `log_s`: log survival curve `log P(no cross by t)`.

## Background

Schiffels and Durbin (2014) introduced the cross-coalescence rate for two
populations as a time-dependent coalescence hazard for lineages sampled from
different groups. Intuitively, it quantifies how quickly lineages from the two
groups find common ancestry as you go back in time, and is widely used as a
measure of divergence and gene flow.

Here, CCR generalizes that idea to arbitrary demographies and sampling schemes
via a simple thought experiment: we imagine each population is colored either
red or blue, and we tag lineages accordingly. We then track the joint process of
red and blue lineages across multiple demes and define the curve as the
instantaneous hazard of the first red-blue coalescence event. This reduces to
the Schiffels-Durbin CCR in the two-sample, two-population setting, while
supporting more complex graphs, time-varying sizes, and migration histories.

## Usage

```python
import jax.numpy as jnp
import stdpopsim

from demestats.ccr import CCRCurve, CCRMeanFieldCurve

demo = stdpopsim.IsolationWithMigration(
    NA=5000, N1=4000, N2=1000, T=1000, M12=1e-4, M21=2e-4
).model.to_demes()

t = jnp.linspace(0.0, 2000.0, 200)
num_samples = {"pop1": (1, 0), "pop2": (0, 1)}

exact = CCRCurve(demo, k=2)(t=t, num_samples=num_samples, params={})
mf = CCRMeanFieldCurve(demo, k=2)(t=t, num_samples=num_samples, params={})
```

## Exact vs mean-field

Exact CCR (`CCRCurve`) tracks the full colored lineage-count CTMC. The state
space grows as `(k+1)^(2d)` for `k` total samples and `d` demes, which becomes
intractable quickly. The implementation guards against this with
`DEMESTATS_CCR_MAX_STATES` and will error if the state space is too large.

Mean-field CCR (`CCRMeanFieldCurve`) evolves only the expected red/blue counts
per deme using a deterministic ODE. This is much faster and scales to large
`k` and `d`, but is an approximation.

## Numerical comparison (IWM example)

The following example compares the curves on a standard isolation-with-migration
demography. In practice, the mean-field approximation tracks the exact curve
closely for typical settings.

```python
import numpy as np

rel_err = np.max(
    np.abs(np.asarray(mf["c"]) - np.asarray(exact["c"]))
    / np.maximum(np.asarray(exact["c"]), 1e-12)
)
print("max relative error in c:", rel_err)
```

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6.0, 3.5))
ax.plot(t, exact["c"], label="exact CCR", lw=2)
ax.plot(t, mf["c"], label="mean-field CCR", lw=2, linestyle="--")
ax.set_xlabel("time")
ax.set_ylabel("c(t)")
ax.set_title("CCR: exact vs mean-field (IWM)")
ax.legend(frameon=False)
fig.tight_layout()
```

## Power to detect recent migration

The mean field CCR curve can be used to infer very recent migration (e.g., within the last 20 generations) when using a large sample size ($k=100$). The following example demonstrates this power by comparing two IWM models: one with continuous migration until the present, and another where migration ceases 20 generations ago.

```python
import demes
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from demestats.ccr import CCRMeanFieldCurve

def make_model(migration_end_time=0):
    b = demes.Builder(description="YRI-CEU like IWM")
    b.add_deme("ancestral", epochs=[dict(start_size=15000, end_time=2500)])
    b.add_deme("YRI", ancestors=["ancestral"], epochs=[dict(start_size=20000)])
    b.add_deme("CEU", ancestors=["ancestral"], epochs=[dict(start_size=10000)])
    
    # Symmetric migration rate of 5e-5
    b.add_migration(
        demes=["YRI", "CEU"], 
        rate=5e-5, 
        start_time=2500, 
        end_time=migration_end_time
    )
    return b.resolve()

# Parameters
t_max = 200
t = jnp.linspace(0.0, t_max, 200)

# Parameters
t_max = 1000
t = jnp.linspace(0.0, t_max, 200)

ks = [1, 5, 20, 100, 200]
print(f"Comparing models with ks={ks}...")

# Compute curves for both models for each k
fig, ax = plt.subplots(figsize=(10, 7))
cmap = plt.get_cmap('viridis')
# Avoid darkest/lightest ends if desired, or just use linear
colors = cmap(np.linspace(0, 0.9, len(ks)))

for i, k in enumerate(ks):
    num_samples = {"YRI": (k, 0), "CEU": (0, k)}
    
    # Model 1: Continuous Migration
    graph_cont = make_model(migration_end_time=0)
    mf_cont = CCRMeanFieldCurve(graph_cont, k=2*k)(t=t, num_samples=num_samples)
    
    # Model 2: Truncated Migration
    graph_trunc = make_model(migration_end_time=20)
    mf_trunc = CCRMeanFieldCurve(graph_trunc, k=2*k)(t=t, num_samples=num_samples)
    
    # Density f(t) = rate(t) * S(t)
    dens_cont = mf_cont["c"] * jnp.exp(mf_cont["log_s"])
    dens_trunc = mf_trunc["c"] * jnp.exp(mf_trunc["log_s"])
    
    color = colors[i]
    ax.plot(t, dens_cont, label=f"k={k}", color=color, lw=2)
    
    # Only plot truncated migration for t >= 20 (where density > 0)
    mask = t >= 20
    ax.plot(t[mask], dens_trunc[mask], color=color, linestyle="--", lw=1.5, alpha=0.7)

ax.set_title("Resulting Coalescent Density (Solid=Continuous, Dashed=Truncated)")
ax.set_xlabel("Generations ago")
ax.set_ylabel("Cross-Coalescence Density (log scale)")
#ax.set_yscale('log')
ax.axvline(20, color='gray', linestyle=':', alpha=0.5, label="t=20 cutoff")

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color=colors[i], label=f'k={k}') for i, k in enumerate(ks)]
legend_elements.append(Line2D([0], [0], color='black', linestyle='-', label='Continuous'))
legend_elements.append(Line2D([0], [0], color='black', linestyle='--', label='Truncated'))

ax.legend(handles=legend_elements)
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.show()
```

The plot shows the cross-coalescence density for sample sizes $k \in \{1, 5, 20, 100, 200\}$ on a log scale. The dashed lines represent the truncated migration model, where the probability of recent cross-coalescence drops effectively to zero (falling off the log scale) for $t < 20$. The solid lines show the continuous migration model. For smaller $k$, the density is low and the distinction between models is less pronounced in magnitude. However, as $k$ increases, the expected density of cross-coalescence events in the recent past rises significantly, providing a strong, distinguishable signal that allows us to reject the truncated migration model.

## Real data analysis

The following example demonstrates how to compute the CCR on a real dataset.
We use the tree sequence from the "Unified Genomes" dataset (HGDP + 1kG + SGDP + Ancients)
restricted to chromosome 20. We compute the minimum cross-coalescence time between
YRI (Yoruba in Ibadan, Nigeria) and CEU (Utah Residents (CEPH) with Northern and Western European Ancestry)
populations.

```python
import tskit
import numpy as np
import matplotlib.pyplot as plt

def get_min_cross_coalescence_time(tree, samples1, samples2):
    """
    Computes the minimum time to the most recent common ancestor (TMRCA)
    between any lineage in samples1 and any lineage in samples2 in the given tree.
    """
    # 1. Collect all ancestors of samples1
    ancestors1 = set()
    current_nodes = set(samples1)
    
    while current_nodes:
        ancestors1.update(current_nodes)
        next_nodes = set()
        for u in current_nodes:
            p = tree.parent(u)
            if p != tskit.NULL and p not in ancestors1:
                next_nodes.add(p)
        current_nodes = next_nodes
        
    # 2. Traverse up from samples2, finding the minimum time of intersection
    min_time = np.inf
    current_nodes = set(samples2)
    visited2 = set() 
    
    while current_nodes:
        next_nodes = set()
        for u in current_nodes:
            if u in ancestors1:
                t = tree.time(u)
                if t < min_time:
                    min_time = t
                # Any ancestor of u has time > t, so we don't need to continue up from here
                # to find a *lower* coalescence time.
            else:
                p = tree.parent(u)
                if p != tskit.NULL and p not in visited2:
                    next_nodes.add(p)
                    visited2.add(p)
        
        current_nodes = next_nodes
        
        # Optimization: stop if all next nodes are older than current min_time
        if min_time != np.inf:
             if not next_nodes:
                 break
             min_next_time = min(tree.time(x) for x in next_nodes)
             if min_next_time > min_time:
                 break
                 
    return min_time

# Load the tree sequence
try:
    ts = tskit.load("/scratch/unified/hgdp_tgp_sgdp_high_cov_ancients_chr20_p.dated.trees")
    print(f"Loaded tree sequence with {ts.num_trees} trees and {ts.num_samples} samples")

    # Helper to find population IDs
    def get_pop_id(name):
        pops_iter = ts.populations() if callable(ts.populations) else ts.populations
        for pop in pops_iter:
            if pop.metadata:
                try:
                    import json
                    md = json.loads(pop.metadata.decode('utf-8')) if isinstance(pop.metadata, bytes) else pop.metadata
                    if md.get('name') == name:
                        return pop.id
                except:
                    continue
        return None

    pop1 = get_pop_id("YRI")
    pop2 = get_pop_id("CEU")

    if pop1 is not None and pop2 is not None:
        samples1 = ts.samples(population=pop1)
        samples2 = ts.samples(population=pop2)
        print(f"Comparing YRI ({len(samples1)} samples) vs CEU ({len(samples2)} samples)")

        times = []
        # Downsample to ~200 trees uniformly across the sequence to balance resolution and linkage
        target_trees = 200
        step = max(1, int(ts.num_trees / target_trees))
        
        for i, tree in enumerate(ts.trees()):
            if i % step == 0:
                t_ccr = get_min_cross_coalescence_time(tree, samples1, samples2)
                times.append(t_ccr)
        
        print(f"Computed CCR for {len(times)} trees.")
        
        # Plot Empirical CCR (CDF)
        times = np.sort(times)
        cdf = np.arange(1, len(times) + 1) / len(times)

        plt.figure(figsize=(8, 5))
        plt.step(times, cdf, where='post', label="Empirical CCR")
        plt.xlabel("Generations ago")
        plt.ylabel("Cumulative Probability")
        plt.title("Empirical CCR: YRI vs CEU (Chr20)")
        plt.grid(True, alpha=0.3)
        plt.show()

    else:
        print("Populations not found.")

except Exception as e:
    print(f"Could not load data or run analysis: {e}")
```

```

## Fitting Demographic Parameters

We can now use the empirical CCR curve derived from the real data to fit demographic parameters. Here, we estimate the **recent exponential growth rate** of the CEU population and the **symmetric migration rate** between YRI and CEU. We use `scipy.optimize` to minimize the mean squared error between the empirical CCR and the Mean-Field model prediction.

```{code-cell} python
from scipy.optimize import minimize
import demes
from demestats.ccr import CCRMeanFieldCurve

# Ensure we have the data from the previous step
if 'times' not in locals() or 'samples1' not in locals():
    print("Please run the 'Real data analysis' cell first.")
else:
    empirical_times = jnp.sort(jnp.array(times))
    empirical_cdf = jnp.arange(1, len(empirical_times) + 1) / len(empirical_times)
    
    # Sample sizes from the real data
    n_yri = len(samples1)
    n_ceu = len(samples2)
    k_total = n_yri + n_ceu
    num_samples = {"YRI": (n_yri, 0), "CEU": (0, n_ceu)}

    def make_parametric_model(r, m):
        # r: growth rate for CEU (positive = growing forward in time)
        # m: symmetric migration rate
        # Fixed parameters based on IWM / literature
        N_YRI = 20000
        N0_CEU = 30000 # Present day effective size approximation
        
        b = demes.Builder(description="Parametric Fit")
        b.add_deme("ancestral", epochs=[dict(start_size=15000, end_time=2500)])
        b.add_deme("YRI", ancestors=["ancestral"], epochs=[dict(start_size=N_YRI)])
        
        # CEU: Exponential growth in the recent epoch (from 2500 gens to present)
        # N(t_ago) = N0 * exp(-r * t_ago)
        start_size_ceu = max(100, N0_CEU * np.exp(-r * 2500))
        b.add_deme("CEU", ancestors=["ancestral"], epochs=[dict(start_size=start_size_ceu, end_size=N0_CEU)])
        
        b.add_migration(demes=["YRI", "CEU"], rate=m, start_time=2500, end_time=0)
        return b.resolve()

    def loss_func(params):
        r, log_m = params
        m = np.exp(log_m)
        
        # Bounds check / Prior
        if r < -0.01 or r > 0.1: return 1e9
        if m < 1e-8 or m > 1e-2: return 1e9
        
        try:
            graph = make_parametric_model(r, m)
            # Compute model CCR at exact empirical event times
            mf = CCRMeanFieldCurve(graph, k=k_total)(t=empirical_times, num_samples=num_samples)
            # CDF = 1 - S(t) = 1 - exp(log_s)
            model_cdf = 1.0 - jnp.exp(mf["log_s"])
            
            # MSE Loss
            mse = jnp.mean((empirical_cdf - model_cdf)**2)
            return float(mse)
        except Exception:
            return 1e9

    print("Fitting parameters (this may take a minute)...")
    # Initial guess: r=0.001, m=1e-5
    x0 = [0.001, np.log(1e-5)]
    
    # Use Nelder-Mead for robustness
    res = minimize(loss_func, x0, method='Nelder-Mead', tol=1e-4, options={'maxiter': 100, 'disp': True})
    
    est_r = res.x[0]
    est_m = np.exp(res.x[1])
    
    print(f"Estimated Growth Rate (r): {est_r:.5f}")
    print(f"Estimated Migration Rate (m): {est_m:.2e}")
    
    # Plotting result
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Empirical
    ax.step(empirical_times, empirical_cdf, where='post', label="Empirical CCR", color='black', alpha=0.6)
    
    # Best Fit Model curve (on finer grid for visualization)
    t_plot = jnp.linspace(0, max(empirical_times)*1.1, 200)
    best_graph = make_parametric_model(est_r, est_m)
    mf_fit = CCRMeanFieldCurve(best_graph, k=k_total)(t=t_plot, num_samples=num_samples)
    cdf_fit = 1.0 - jnp.exp(mf_fit["log_s"])
    
    ax.plot(t_plot, cdf_fit, label=f"Best Fit (r={est_r:.4f}, m={est_m:.2e})", color='red', lw=2)
    
    ax.set_title("CCR Parameter Inference: YRI-CEU")
    ax.set_xlabel("Generations ago")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
```
