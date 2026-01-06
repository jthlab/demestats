# momi3 tutorial (within demestats)

This tutorial is a self-contained introduction to **momi3**, implemented as part of the
`demestats` package (specifically the `demestats.sfs` modules). `demestats` also includes
other components (IICR/CCR curves, event trees, constraints, etc.), but this guide focuses
only on the SFS-based inference workflow that people refer to as *momi3*.

The corresponding Jupyter notebook is available at `docs/momi3_tutorial.ipynb`.

## Overview

The momi3 workflow inside `demestats` consists of:

1. Simulating (or loading) genetic data.
2. Computing an allele-frequency spectrum (AFS).
3. Building an `ExpectedSFS` model from a demes graph.
4. Evaluating SFS log-likelihoods.
5. (Optionally) optimizing demographic parameters with constraints.

## Simulation

We will simulate a simple isolation-with-migration (IWM) model with two populations.
This uses `msprime` to build a demography and simulate ancestry/mutations.

```python
import msprime as msp
import demesdraw

demo = msp.Demography()
demo.add_population(initial_size=5000, name="anc")
demo.add_population(initial_size=5000, name="P0")
demo.add_population(initial_size=5000, name="P1")
demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
demo.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")

g = demo.to_demes()
demesdraw.tubes(g)
```

Simulate ancestry and mutations:

```python
sample_size = 10
samples = {"P0": sample_size, "P1": sample_size}

anc = msp.sim_ancestry(
    samples=samples,
    demography=demo,
    recombination_rate=1e-8,
    sequence_length=1e8,
    random_seed=12,
)
ts = msp.sim_mutations(anc, rate=1e-8, random_seed=13)
```

Compute the AFS (allele-frequency spectrum):

```python
afs_samples = {"P0": sample_size * 2, "P1": sample_size * 2}
afs = ts.allele_frequency_spectrum(
    sample_sets=[ts.samples([0]), ts.samples([1])],
    span_normalise=False,
)
```

## ExpectedSFS (momi3 core)

The `ExpectedSFS` object is the core momi3 component. It maps a demes graph and sample
configuration to the expected spectrum under a demographic model.

```python
from demestats.sfs import ExpectedSFS

esfs = ExpectedSFS(g, num_samples=afs_samples)
expected = esfs(params={})
```

## SFS log-likelihood

For likelihood-based inference, use the SFS log-likelihood helpers from
`demestats.loglik.sfs_loglik`.

```python
from demestats.loglik.sfs_loglik import sfs_loglik

ll = sfs_loglik(
    afs=afs,
    expected_sfs=expected,
    sequence_length=1e8,
    theta=1e-8,
)
```

## Parameterization and constraints

`demestats` automatically generates parameter constraints for a given model via
`EventTree` and `constraints_for`. This is part of the momi3 workflow because it defines
the feasible parameter space for SFS-based optimization.

```python
from demestats.constr import EventTree, constraints_for

et = EventTree(g)
variables = et.variables

cons = constraints_for(et, *variables)
A_eq, b_eq = cons["eq"]
A_ineq, b_ineq = cons["ineq"]
```

## Putting it together (minimal optimization sketch)

A full optimizer is not shown here, but the typical flow is:

1. Choose a parameter vector (subset of `et.variables`).
2. Use `constraints_for` to get linear constraints.
3. Convert parameter vectors into demes parameters.
4. Evaluate SFS log-likelihood and optimize.

If you want a complete worked example, use the notebook at
`docs/momi3_tutorial.ipynb`.

## Where to go next

- For other `demestats` features (IICR/CCR curves, event trees, etc.), see the main
  documentation sections in `docs/`.
- For API details, see the generated module reference under `docs/api`.
