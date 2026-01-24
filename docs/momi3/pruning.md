# Pruning (manual downsampling)

This tutorial describes the pruning option for `ExpectedSFS`. Pruning lets you
insert explicit downsampling events into the event tree to keep the internal
state size small when few lineages are likely to be ancestral.

## Basic usage

`prune` is optional and can be either a mapping or a list of tuples.

```python
from demestats.sfs import ExpectedSFS

esfs = ExpectedSFS(
    demo,
    num_samples={"A": 12, "B": 8},
    prune={"A": 4},
)
```

This inserts a downsample event directly above the leaf for deme `A`.

You can also specify an explicit location:

```python
esfs = ExpectedSFS(
    demo,
    num_samples={"A": 12, "B": 8},
    prune=[("A", 4, ("demes", 0, "epochs", 0, "end_time"))],
)
```

Supported forms for `prune` entries:

- `{"A": 4}` means prune `A` to 4 at the leaf.
- `[("A", 4)]` is the same as the mapping form.
- `[("A", 4, at)]` inserts at a specific location.

## Targeting a location

`at` can be either:

- A node id from the event tree, or
- A demes time path (for example `("demes", 1, "start_time")`).

If you pass a time path:

- If exactly one node at that time has `A` in its block, pruning inserts above
  that node.
- Otherwise, if exactly one edge has parent time `at` and a child block
  containing `A`, pruning inserts along that edge.
- If neither case is unique, an error is raised to avoid ambiguous edits.

To discover node ids, inspect the event tree:

```python
from demestats.sfs import ExpectedSFS

esfs = ExpectedSFS(demo, num_samples={"A": 12, "B": 8})
for node in esfs.et.nodes:
    attrs = esfs.et.nodes[node]
    print(node, attrs.get("event"), attrs.get("t"), attrs.get("block"))
```

## What pruning does (and does not) change

- The SFS output shape is still determined by `num_samples`.
- Pruning only changes the internal state during traversal.
- `m == n` is a no-op (no change).
- `m == 0` produces an all-zero SFS.

Pruning uses a hypergeometric downsample along the specified axis. This is an
approximation that is useful for dimension reduction, but it does not always
match the exact smaller-sample model.

### Example: deep ancestry

If a split is very old, at most one lineage typically survives to the merge.
Pruning to 1 at that merge time often has no effect on the SFS:

```python
import demes
from demestats.sfs import ExpectedSFS

b = demes.Builder()
b.add_deme("ANC", epochs=[{"end_time": 50.0, "start_size": 1.0}])
b.add_deme("A", ancestors=["ANC"], epochs=[{"start_size": 1.0}])
g = b.resolve()

esfs = ExpectedSFS(g, num_samples={"A": 4})
base = esfs()

pruned = ExpectedSFS(
    g,
    num_samples={"A": 4},
    prune=[("A", 1, ("demes", 1, "start_time"))],
)()
```

In this setup, `base` and `pruned` agree.

## Important caveats

- Pruning after a merge acts on a mixed axis (lineages from multiple demes).
  Even if an event is a no-op (for example, a pulse with p=0), pruning a mixed
  axis is not equivalent to pruning before the merge.
- If your model includes continuous migration, the SFS code assumes sample
  sizes of at least 4 in populations participating in migration. Pruning below
  that may lead to errors or poor approximations.

## Recommended workflow

1. Start with a working model without pruning.
2. Insert pruning events in obvious bottlenecks.
3. Validate by comparing against the unpruned model on small sample sizes.
4. Increase sample sizes and add pruning as needed.
