# Time-Stratified SFS

`demestats.tsfs` extends the expected SFS calculation to a fixed time grid. Instead of
returning one expected count for each derived-allele configuration, it partitions
that expected count by mutation age.

Given a grid

`0 = t[0] < t[1] < ... < t[M] = inf`,

the entry `tsfs[i, ...]` is the expected number of mutations that:

- arose during the time interval `[t[i], t[i + 1])`, and
- subtend the corresponding lineage configuration on the remaining axes.

Summing over the first axis recovers the ordinary expected SFS.

## Basic usage

```python
import jax.numpy as jnp
from demestats.sfs import ExpectedSFS
from demestats.tsfs import ExpectedTSFS

time_grid = [0.0, 50.0, 200.0, jnp.inf]

etsfs = ExpectedTSFS(
    demo,
    num_samples={"A": 4, "B": 4},
    time_grid=time_grid,
)

tsfs = etsfs()
print(tsfs.shape)
# (3, 5, 5)

esfs = ExpectedSFS(demo, num_samples={"A": 4, "B": 4})()
print(jnp.allclose(tsfs.sum(axis=0), esfs))
# True
```

The first axis indexes time bins. The remaining axes use the same convention as
`ExpectedSFS`: for a deme with `n` haploid samples, the corresponding axis has
length `n + 1`.

## Time grid requirements

`time_grid` is fixed when the `ExpectedTSFS` object is created. It must:

- be one-dimensional,
- be strictly increasing,
- start at `0`, and
- end at `inf`.

The grid is interpreted in the same time units as the input `demes.Graph`. Internal
rescaling is handled automatically, so you should always specify the grid in the
original demographic-model units.

## Output interpretation

If `num_samples={"A": nA, "B": nB}`, then:

- `tsfs.shape == (M, nA + 1, nB + 1)`,
- `tsfs[i, j, k]` is the expected number of mutations from time bin `i` with
  `j` derived alleles in `A` and `k` derived alleles in `B`,
- `tsfs.sum(axis=0)` is the ordinary expected SFS, and
- `tsfs[:, 0, 0]` and `tsfs[:, -1, -1]` are zero, just as in the ordinary SFS.

The total expected number of mutations in a time interval is `tsfs[i].sum()`.

## Parameter overrides

`ExpectedTSFS` accepts the same parameter override dictionary as `ExpectedSFS`.
This makes it easy to evaluate the TSFS across different demographic parameter
settings without rebuilding the object:

```python
etsfs = ExpectedTSFS(demo, num_samples={"A": 4}, time_grid=[0.0, 100.0, jnp.inf])

params = {
    etsfs.variable_for(("demes", 0, "epochs", 0, "start_size")): 2_000.0,
}

tsfs = etsfs(params)
```

## Pruning

`ExpectedTSFS` supports the same `prune=` option as `ExpectedSFS`. Pruning only
changes the internal dynamic-programming state. It does not change the output
shape.

```python
from demestats.tsfs import ExpectedTSFS

etsfs = ExpectedTSFS(
    demo,
    num_samples={"A": 12, "B": 8},
    time_grid=[0.0, 100.0, 500.0, jnp.inf],
    prune={"A": 4},
)
```

For the pruning syntax and caveats, see [``Pruning``](pruning.md).

## Current scope

The first version of `demestats.tsfs` focuses on direct TSFS evaluation through
`ExpectedTSFS(...)(params)`. The `ExpectedSFS.tensor_prod` helper does not yet
have a TSFS counterpart.

For API details, see the generated reference under `API`.
