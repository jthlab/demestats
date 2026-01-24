---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
---

# Key concepts

This page defines the core concepts in `demestats`. We use a
single running example throughout so the terminology stays concrete.

## Running example (two-population IWM with a pulse)

We define a simple isolation-with-migration (IWM) model plus a pulse. This one model is
used for all examples below.

```python
import msprime as msp
import demesdraw

# Two populations with symmetric migration.
demo = msp.Demography()
demo.add_population(initial_size=5000, name="anc")
demo.add_population(initial_size=5000, name="P0")
demo.add_population(initial_size=5000, name="P1")
# Add a pulse (one-time admixture).
demo.add_mass_migration(time=200, source="P0", dest="P1", proportion=0.1) 
demo.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")
demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)

# Convert to a demes graph and visualize.
g = demo.to_demes()
demesdraw.tubes(g)
```

## Demography and demes graph

Demography is the user-facing description of population history (sizes, splits,
migration, pulses). `demestats` expects a [demes](https://popsim-consortium.github.io/demes-docs/latest/introduction.html)-formatted model (e.g. a `demes.Graph` object).

In our example:
- `demo` is a `msprime.Demography`.
- `g = demo.to_demes()` is a `demes.Graph`.

The `demes.Graph` is the input to functions in `demestats`.

## Paths

A path is a tuple of strings/integers that identifies a specific parameter in the
nested dictionary representation of a `demes.Graph` model. Paths in `demestats` look like:

```python
("demes", 0, "epochs", 0, "end_size")
("migrations", 0, "rate")
("pulses", 0, "time")
```
 
For example, to access the ending size of the ancestral population "anc":

- `g.asdict()['demes'][0]['epochs'][0]['end_size']` is a sequence of keys for `demes.Graph`
- `("demes", 0, "epochs", 0, "end_size")` is a tuple path for `demestats`

To access the rate of migration from "P0" to "P1":

- `g.asdict()['demes']['migrations'][0]['rate']` is a sequence of keys for `demes.Graph`
- `("migrations", 0, "rate")` is a tuple path for `demestats`

Paths are the raw coordinates that variables and constraints are built from.

## Event tree

The event tree is the internal probabilistic graphical model used by `demestats` to
perform computations (SFS, likelihoods, IICR curves). It is derived from a `demes.Graph` object that the user defines.

```python
from demestats.event_tree import EventTree

et = EventTree(g)
```

Notes:

- Time ordering is enforced. Events appear in the tree in the order implied by the
  demography.
- Topology is fixed by the demography. If you change the demography in a way that
  changes the ordering or branching structure of events, you must use the new `demes.Graph` to build a new event tree.

## Variables

A variable is an optimizable parameter derived from one or more paths. Some paths
are grouped together because they are equal *by construction* in the base demography.
This grouping shows up as `frozenset` entries.

```python
vars_ = et.variables
vars_[:5]
```

Expected output:

```python
[frozenset({('demes', 0, 'epochs', 0, 'end_size'),
            ('demes', 0, 'epochs', 0, 'start_size')}),
 frozenset({('demes', 1, 'epochs', 0, 'end_size'),
            ('demes', 1, 'epochs', 0, 'start_size')}),
 frozenset({('demes', 2, 'epochs', 0, 'end_size'),
            ('demes', 2, 'epochs', 0, 'start_size')}),
 frozenset({('demes', 1, 'proportions', 0)}),
 frozenset({('demes', 2, 'proportions', 0)})]
```

### Example: grouped variables

- Constant sizes: The base demography `g` constructed has constant population sizes. `("demes", 0, "epochs", 0, "start_size")` and `("demes", 0, "epochs", 0, "end_size")` are tied so they fall inside the same `frozenset` object. 
- A split time and a migration start time may be tied if they are the same event in the
  base demography. e.g. `frozenset({('demes', 0, 'epochs', 0, 'end_time'), ('demes', 1, 'start_time'), ('demes', 2, 'start_time'), ('migrations', 0, 'start_time'), ('migrations', 1, 'start_time')})`.

To see an examples of how to modify `frozenset` objects please refer to [`Special Examples`](momi3/special_examples.md).

### Same time vs same variable

Events on different branches can share the same numerical time but still be distinct
variables. They are only tied when the base demography identifies them as the same event. For example, `frozenset({('migrations', 0, 'rate')})` and `frozenset({('migrations', 1, 'rate')})` are distinct variables.

## Constraints

Constraints encode valid parameter ranges and event ordering for the current
event tree. They are derived from the demography and the variable grouping.

```python
from demestats.constr import constraints_for

cons = constraints_for(et, *et.variables)
A_eq, b_eq = cons["eq"]
A_ineq, b_ineq = cons["ineq"]
```

Typical constraint types:

- Nonnegativity of times and sizes.
- Upper bounds (e.g., migration rates in [0, 1]).
- Ordering constraints (e.g., split time precedes the present).

If you change the demography in a way that changes event ordering, you must rebuild the
event tree and constraints.

Please first refer to [`momi3 Tutorial`](momi3/momi3_tutorial.md) or `IICR Tutorial` and then [`Model Constraints`](momi3/model_constraints.md) to understand how to modify the constraints to one's needs.

## Parameter overrides

Most `demestats` APIs accept parameter overrides as a dictionary mapping variables
(or paths) to numeric values. This is how you evaluate models at specific parameter
settings.

```python
from demestats.event_tree import EventTree
et = EventTree(g)

# Pick variables (by path) from the event tree.
v_split = et.variable_for(("demes", 0, "epochs", 0, "end_time"))
v_mig = et.variable_for(("migrations", 0, "rate"))

params = {
    v_split: 1200.0,
    v_mig: 2e-4,
}
```

The `params` dict can then be passed into objects like `ExpectedSFS` or `IICRCurve`. Please refer to the tutorials to see examples.

### How overrides relate to constraints

- The constraints define the valid region for variables.
- The parameter overrides must respect those constraints.

If you supply a parameter set that violates constraints (e.g., negative times or an
ordering violation), the model is invalid for that event tree. In practice, this
means you should only evaluate parameter overrides that satisfy the constraint system.

## Concept map (summary)

- Demography (user input) → demes.Graph (`g`)
- demes.Graph → EventTree (`et`)
- EventTree → variables (`et.variables`) and constraints (`constraints_for`)
- Parameter overrides: `dict[variable -> value]` used to evaluate the model
- If topology or ordering changes: rebuild the event tree and constraints
