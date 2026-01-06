---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
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

```{code-cell} python
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

```{code-cell} python
import numpy as np

rel_err = np.max(
    np.abs(np.asarray(mf["c"]) - np.asarray(exact["c"]))
    / np.maximum(np.asarray(exact["c"]), 1e-12)
)
print("max relative error in c:", rel_err)
```

```{code-cell} python
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
