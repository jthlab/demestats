---
jupytext:
  formats: ipynb,md:myst,py:percent
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
---

## Simulation

This notebook is a practical walk-through of the SFS-based workflow often called momi3,
implemented inside the `demestats.sfs` modules. demestats also includes other tools
(IICR/CCR curves, event trees, constraints, etc.), but we focus only on the SFS
inference path here.

We start by simulating a simple isolation-with-migration model and producing an AFS.


```python
# imports

# build IWM demography
import demesdraw
import msprime as msp

demo = msp.Demography()
demo.add_population(initial_size=5000, name="anc")
demo.add_population(initial_size=5000, name="P0")
demo.add_population(initial_size=5000, name="P1")
demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
tmp = [f"P{i}" for i in range(2)]

# split time
demo.add_population_split(time=1000, derived=tmp, ancestral="anc")

# visualize
g = demo.to_demes()
demesdraw.tubes(g)

# simulate ancestry + mutations
sample_size = 10
samples = {f"P{i}": sample_size for i in range(2)}
anc = msp.sim_ancestry(
    samples=samples,
    demography=demo,
    recombination_rate=1e-8,
    sequence_length=1e8,
    random_seed=12,
)
ts = msp.sim_mutations(anc, rate=1e-8, random_seed=13)

# compute AFS
afs_samples = {f"P{i}": sample_size * 2 for i in range(2)}
afs = ts.allele_frequency_spectrum(
    sample_sets=[ts.samples([1]), ts.samples([2])],
    span_normalise=False,
)

```

## Demographic parameters in momi3

momi3 treats a demography as a set of variables defined by an event tree. Each variable
corresponds to one or more paths in the demes model. This section inspects which
parameters are available for inference in a simple model.


```python
# show parameter variables
from demestats.constr import EventTree, constraints_for

et = EventTree(g)
et.variables

```

## Demographic constraints in momi3

Constraints encode biologically valid parameter ranges and dependencies (e.g., times
must be nonnegative and event times must align). We use `constraints_for` to obtain
linear equality/inequality constraints for optimization.


```python
# pick three parameters: anc size, P0->P1 rate, split time
triplet = [
    frozenset(
        {("demes", 0, "epochs", 0, "end_size"), ("demes", 0, "epochs", 0, "start_size")}
    ),
    ("migrations", 0, "rate"),
    frozenset(
        {
            ("demes", 0, "epochs", 0, "end_time"),
            ("demes", 1, "start_time"),
            ("demes", 2, "start_time"),
            ("migrations", 0, "start_time"),
            ("migrations", 1, "start_time"),
        }
    ),
]

cs = constraints_for(et, *triplet)
cs

```

## Modifying the constraints

You can impose additional constraints that reflect modeling choices. A common example is
symmetry of migration rates. Here we add a custom equality rule to tie two rates together.


```python
# enforce symmetric migration by adding an equality row

import numpy as np

sel = [
    frozenset(
        {("demes", 0, "epochs", 0, "end_size"), ("demes", 0, "epochs", 0, "start_size")}
    ),
    ("migrations", 0, "rate"),
    ("migrations", 1, "rate"),
]
constraint = constraints_for(et, *sel)

A_eq, b_eq = constraint["eq"]
i0rate, i1rate = 1, 2

# add: rate_0 - rate_1 = 0
new_rule = np.zeros((1, A_eq.shape[1] if A_eq.size else len(sel)))
new_rule[0, i0rate] = 1.0
new_rule[0, i1rate] = -1.0

A_eq = np.vstack([A_eq, new_rule]) if A_eq.size else new_rule
b_eq = np.concatenate([b_eq, [0.0]]) if b_eq.size else np.array([0.0])
constraint["eq"] = (A_eq, b_eq)

constraint

```

## Inference using SFS-based methods in momi3

The core object is `ExpectedSFS`, which maps a demes graph and sample configuration to
an expected spectrum. We then evaluate log-likelihoods using helpers in
`demestats.loglik.sfs_loglik`.


```python
# log-likelihood and finite-difference gradient for one parameter

import numpy as np

from demestats.loglik.sfs_loglik import sfs_loglik
from demestats.sfs import ExpectedSFS

param_key = ("migrations", 0, "rate")
ESFS = ExpectedSFS(g, num_samples={"P0": 4, "P1": 4})
afs = np.array([0, 12, 7, 3, 0], dtype=float)


def ll_at(val):
    params = {param_key: float(val)}
    esfs = np.asarray(ESFS(params))
    return float(sfs_loglik(afs, esfs))


def grad_fd(val, h=1e-6):
    return (ll_at(val + h) - ll_at(val - h)) / (2.0 * h)


val = 1e-4
print("loglik =", ll_at(val))
print("grad   =", grad_fd(val))

```

## Likelihood scan helper

To build intuition, we can scan a single parameter across a grid and compute the
log-likelihood surface. This helps verify that the model behaves as expected before
running full optimization.


```python
# plot negative log-likelihood across a grid

from typing import Dict, List, Sequence

import jax.numpy as jnp
from jax import lax

Var = object


def _vec_to_dict_jax(v: jnp.ndarray, keys: Sequence[Var]) -> Dict[Var, jnp.ndarray]:
    return {k: v[i] for i, k in enumerate(keys)}


def plot_sfs_likelihood(
    demo, paths, vec_values, afs, afs_samples, theta=None, sequence_length=None
):
    import matplotlib.pyplot as plt

    from demestats.loglik.sfs_loglik import sfs_loglik
    from demestats.sfs import ExpectedSFS

    path_order: List[Var] = list(paths)
    esfs = ExpectedSFS(demo, num_samples=afs_samples)

    def evaluate_at_vec(vec):
        vec_array = jnp.atleast_1d(vec)
        params = _vec_to_dict_jax(vec_array, path_order)
        e1 = esfs(params)
        return -sfs_loglik(afs, e1, sequence_length, theta)

    results = lax.map(evaluate_at_vec, vec_values)

    plt.figure(figsize=(10, 6))
    plt.plot(vec_values, results, "r-", linewidth=2)
    plt.xlabel("Parameter value")
    plt.ylabel("Negative Log-Likelihood")
    plt.title("SFS Likelihood Landscape")
    plt.grid(True)
    plt.show()
    return results

```

## Grid scans: size, time, migration

We extend the scan idea to multiple parameters to visualize how the likelihood varies
across the parameter space.


```python
# ancestral size scan

import jax.numpy as jnp

paths = {
    frozenset(
        {
            ("demes", 0, "epochs", 0, "end_size"),
            ("demes", 0, "epochs", 0, "start_size"),
        }
    ): 4000.0
}
vec_values = jnp.linspace(4000, 6000, 50)
_ = plot_sfs_likelihood(g, paths, vec_values, afs, {"P0": 4, "P1": 4})

# descendant size (P0)
paths = {
    frozenset(
        {
            ("demes", 1, "epochs", 0, "end_size"),
            ("demes", 1, "epochs", 0, "start_size"),
        }
    ): 4000.0
}
vec_values = jnp.linspace(4000, 6000, 50)
_ = plot_sfs_likelihood(g, paths, vec_values, afs, {"P0": 4, "P1": 4})

# split time
paths = {
    frozenset(
        {
            ("demes", 0, "epochs", 0, "end_time"),
            ("demes", 1, "start_time"),
            ("demes", 2, "start_time"),
            ("migrations", 0, "start_time"),
            ("migrations", 1, "start_time"),
        }
    ): 4000.0
}
vec_values = jnp.linspace(500, 1500, 50)
_ = plot_sfs_likelihood(g, paths, vec_values, afs, {"P0": 4, "P1": 4})

# migration rate
paths = {("migrations", 0, "rate"): 0.0001}
vec_values = jnp.linspace(0.00005, 0.0002, 10)
_ = plot_sfs_likelihood(g, paths, vec_values, afs, {"P0": 4, "P1": 4})

```

## Optimization with Poisson likelihood

This section shows a minimal optimization example using a Poisson likelihood. It is a
simple baseline for comparing with more elaborate inference setups.


```python
# run same scan with Poisson likelihood
theta = 1.25e-8
sequence_length = 1_000_000

paths = {("migrations", 0, "rate"): 0.0001}
vec_values = jnp.linspace(0.00005, 0.0002, 10)
_ = plot_sfs_likelihood(
    g,
    paths,
    vec_values,
    afs,
    {"P0": 4, "P1": 4},
    theta=theta,
    sequence_length=sequence_length,
)

```

## Population size change example

We now consider a model with size changes (and possibly time-varying migration) to show
how the same workflow applies to richer demographies.


```python
# model with size changes and migration rate changes

import numpy as np

from demestats.event_tree import EventTree

demo2 = msp.Demography()
demo2.add_population(initial_size=4000, name="anc")
demo2.add_population(initial_size=500, name="P0", growth_rate=-np.log(3000 / 500) / 66)
demo2.add_population(initial_size=500, name="P1", growth_rate=-np.log(3000 / 500) / 66)
demo2.add_population(initial_size=100, name="P2", growth_rate=-np.log(3000 / 100) / 66)

demo2.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
demo2.set_symmetric_migration_rate(populations=("P1", "P2"), rate=0.0001)

demo2.add_population_parameters_change(
    time=65, population="P0", initial_size=3000, growth_rate=0.0
)
demo2.add_population_parameters_change(
    time=65, population="P1", initial_size=3000, growth_rate=0.0
)
demo2.add_population_parameters_change(
    time=66, population="P2", initial_size=3000, growth_rate=0.0
)

demo2.add_migration_rate_change(time=66, source="P0", dest="P1", rate=0.0005)
demo2.add_migration_rate_change(time=66, source="P1", dest="P0", rate=0.0005)
demo2.add_migration_rate_change(time=66, source="P1", dest="P2", rate=0.0005)
demo2.add_migration_rate_change(time=66, source="P2", dest="P1", rate=0.0005)

demo2.add_population_split(time=5000, derived=["P0", "P1", "P2"], ancestral="anc")

g2 = demo2.to_demes()
demesdraw.tubes(g2, log_time=True)

# inspect variables
et2 = EventTree(g2)
et2.variables

```

## Admixture example

Finally, we demonstrate a model with admixture. The same SFS tools apply, but the
parameter set and constraints become more structured.


```python
# model with one admixed population
from demestats.constr import constraints_for
from demestats.event_tree import EventTree

demo3 = msp.Demography()
demo3.add_population(name="P0", initial_size=5000)
demo3.add_population(name="P1", initial_size=5000)
demo3.add_population(name="ADMIX", initial_size=1000)
demo3.add_population(name="anc", initial_size=5000)

demo3.add_admixture(
    time=500, derived="ADMIX", ancestral=["P0", "P1"], proportions=[0.4, 0.6]
)
demo3.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")

g3 = demo3.to_demes()
demesdraw.tubes(g3)

# list variables and print shapes of equality constraints
et3 = EventTree(g3)
for v in et3.variables:
    print(v)

cs3 = constraints_for(et3, *et3.variables)
A_eq3, b_eq3 = cs3["eq"]
A_eq3.shape, b_eq3.shape

```
