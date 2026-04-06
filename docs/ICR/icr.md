---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# ICR: Exact and Mean-Field Curves

This notebook shows how to compute instantaneous coalescence rate
(ICR) curves for several total sample sizes `k`, using both the exact solver
and the mean-field approximation.

`demestats` returns the coalescence hazard `c(t)` (also known as the ICR) together with the log-survival
curve `log_s(t)`. The IICR described by Mazet et al. (2016) is the reciprocal of that hazard:

`IICR(t) = 1 / c(t)`.

## Overview

- `demestats.iicr.IICRCurve`: exact lineage-count CTMC. Accurate, but the state
  space grows quickly with `k`.
- `demestats.iicr.IICRMeanFieldCurve`: deterministic mean-field approximation.
  Much faster for larger sample sizes.

We will use a simple two-deme isolation-with-migration model, split the samples
evenly across the two present-day demes, and compare the resulting curves.

```{code-cell} ipython3
import demes
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from demestats.iicr import IICRCurve, IICRMeanFieldCurve
```

## Build a small example demography

```{code-cell} ipython3
def make_two_deme_model(N=5_000.0, m=1e-3, split_time=1_000.0) -> demes.Graph:
    b = demes.Builder(description="Two demes with symmetric migration")
    b.add_deme("anc", epochs=[dict(start_size=N, end_time=split_time)])
    b.add_deme("A", ancestors=["anc"], start_time=split_time, epochs=[dict(start_size=N)])
    b.add_deme("B", ancestors=["anc"], start_time=split_time, epochs=[dict(start_size=N)])
    b.add_migration(source="A", dest="B", rate=m, start_time=split_time, end_time=0.0)
    b.add_migration(source="B", dest="A", rate=m, start_time=split_time, end_time=0.0)
    return b.resolve()


demo = make_two_deme_model()
split_time = 1_000.0
demo
```

## Time grid and helpers

We use a geometric time grid so the plot has more resolution in the recent past.
The lower endpoint is positive because `geomspace` does not include zero.

```{code-cell} ipython3
t = jnp.geomspace(1.0, 5_000.0, 250)
small_ks = [2, 4, 8]
all_ks = [2, 4, 8, 16, 32, 64]


def balanced_samples(k: int) -> dict[str, int]:
    return {"A": k // 2, "B": k - k // 2}


def iicr_values(curve_out) -> np.ndarray:
    return 1.0 / np.asarray(curve_out["c"])
```

## Compute exact and mean-field curves

For `k = 2, 4, 8` we compute both the exact curve and the mean-field
approximation. For larger sample sizes, we only use the mean-field method.

```{code-cell} ipython3
exact_curves = {}
mf_curves = {}

for k in small_ks:
    num_samples = balanced_samples(k)
    exact_curves[k] = IICRCurve(demo, k=k)(t=t, num_samples=num_samples, params={})
    mf_curves[k] = IICRMeanFieldCurve(demo, k=k)(t=t, num_samples=num_samples, params={})

for k in all_ks:
    if k not in mf_curves:
        mf_curves[k] = IICRMeanFieldCurve(demo, k=k)(
            t=t, num_samples=balanced_samples(k), params={}
        )
```

## Exact vs mean-field

The mean-field approximation is already quite close for modest sample sizes in
this example.

```{code-cell} ipython3
for k in small_ks:
    exact_iicr = iicr_values(exact_curves[k])
    mf_iicr = iicr_values(mf_curves[k])
    rel_err = np.max(np.abs(mf_iicr - exact_iicr) / np.maximum(exact_iicr, 1e-12))
    print(f"k={k:>2}: max relative error = {rel_err:.2%}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7.0, 4.0))
colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(small_ks)))

for color, k in zip(colors, small_ks):
    ax.plot(t, iicr_values(exact_curves[k]), color=color, lw=2, label=f"exact, k={k}")
    ax.plot(
        t,
        iicr_values(mf_curves[k]),
        color=color,
        lw=2,
        linestyle="--",
        label=f"mean-field, k={k}",
    )

ax.axvline(split_time, color="0.6", linestyle=":", lw=1.5, label="split time")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("time")
ax.set_ylabel("IICR(t)")
ax.set_title("IICR: exact vs mean-field")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
```

## Scaling to larger sample sizes

The exact method becomes expensive quickly as `k` grows, but the mean-field
approximation remains practical. The next plot extends the sample size up to
`k = 64`.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7.0, 4.0))
colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.9, len(all_ks)))

for color, k in zip(colors, all_ks):
    ax.plot(t, iicr_values(mf_curves[k]), color=color, lw=2, label=f"k={k}")

ax.axvline(split_time, color="0.6", linestyle=":", lw=1.5, label="split time")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("time")
ax.set_ylabel("IICR(t)")
ax.set_title("Mean-field IICR across sample sizes")
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
```

As `k` increases, the total coalescence hazard rises because there are more
lineage pairs that can coalesce, so the IICR decreases. For exploratory work on
large samples, `IICRMeanFieldCurve` is usually the right starting point.
