# Model constraints

`demestats` can automatically translate a given demographic model into the precise set of numerical constraints that satisfy model restrictions, such as those governing time intervals, population sizes, and admixture events. This eliminates the tedious and challenging manual derivation of constraints, making constrained optimization more accessible. This pages introduces how to understand and modify the constraints.

The corresponding Jupyter notebook is available at `docs/tutorial_code/examples.ipynb`.

Let's explore the isolation-with-migration (IWM) model:

```python
import msprime as msp
import demesdraw

demo = msp.Demography()
demo.add_population(initial_size=5000, name="anc")
demo.add_population(initial_size=5000, name="P0")
demo.add_population(initial_size=5000, name="P1")
demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
demo.add_population_split(time=1000, derived=[f"P{i}" for i in range(2)], ancestral="anc")
g = demo.to_demes()
demesdraw.tubes(g)
```

To see all the parameters in the IWM model:

```python
from demestats.constr import EventTree
et = EventTree(g)
et.variables
```

Reminder that parameters in `frozenset` objects are defined to be the same event by *construction* of the model. Please refer to [`Notation`](https://demestats.readthedocs.io/en/latest/notation.html).

## Linear constraints in demestats

Suppose we were interested in 3 parameters - the ancestral population size, rate of migration from P0 to P1, and the time of divergence. To output the associated linear constraints, we must input a list of the parameters into ``constraints_for``. 

```python
from demestats.constr import constraints_for
momi3_parameters = [
        frozenset({('demes', 0, 'epochs', 0, 'end_size'),
        ('demes', 0, 'epochs', 0, 'start_size')}), # Ancestral population size
        frozenset({('migrations', 0, 'rate')}), # Rate of migration
        frozenset({('demes', 0, 'epochs', 0, 'end_time'),
        ('demes', 1, 'start_time'),
        ('demes', 2, 'start_time'),
        ('migrations', 0, 'start_time'),
        ('migrations', 1, 'start_time')}) # Time of divergence
        ]
constraint = constraints_for(et, *momi3_parameters)
print(constraint)
```

If one is not familiar with ``frozenset`` parameters, one can optionally use the ``variable_for`` function to take paths and find the associated ``frozenset`` parameter. 

```python
parameters = [
    ('demes', 0, 'epochs', 0, 'end_size'), # The ancestral population size
    ('migrations', 0, 'rate'), # Rate of migration from P0 to P1
    ('demes', 0, 'epochs', 0, 'end_time') # Time of divergence
]

momi3_parameters = [et.variable_for(param) for param in parameters]
constraint = constraints_for(et, *momi3_parameters)
print(constraint)
```

The output of ``constraints_for`` is a dictionary with two keys:

```python
{
'eq': (array([], shape=(0, 3), dtype=float64),
        array([], dtype=float64)),

'ineq': (array([[-1., -0., -0.],
                [-0., -1., -0.],
                [ 0.,  1.,  0.],
                [ 0.,  0., -1.]]),
        array([0., 0., 1., 0.]))
}
```

``"eq"``: linear equality constraints will be a tuple of the form ``(A_eq, b_eq)`` such that ``A_eq @ x = b_eq``.

``"ineq"``: linear inequality constraints will be a tuple of the form ``(A_ineq, b_ineq)`` such that ``A_ineq @ x <= b_ineq``.

`constraints_for` will preserve the ordering of the parameters, so we have:

- Column 0: ancestral population size
- Column 1: migration rate from P0 → P1
- Column 2: time of divergence

**Interpretation of inequality constraints**:

- Row 0: ``[-1., -0., -0.] <= 0`` → -(ancestral population size) <= 0 → ancestral population size ≥ 0
- Row 1: ``[-0., -1., -0.] <= 0`` → migration rate ≥ 0
- Row 2: ``[ 0.,  1.,  0.] <= 1`` → migration rate ≤ 1
- Row 3: ``[ 0.,  0., -1.] <= 0`` → split time ≥ 0

These constraints ensure meaningful parameter ranges: population sizes and times must be nonnegative, and migration rates must lie within the range of ``[0, 1]``.

In general, ``constraints_for`` automatically generates the linear constraints required for optimization. To verify and interpret the constraints more easily, one can optionally use the ``print_constraints`` function.

```python
from demestats.constr import print_constraints
print_constraints(constraint, momi3_parameters)
```

The output:

```python
==================================================
Linear Equalities: Ax = b
==================================================

None

==================================================
Linear Inequalities: Ax <= b
==================================================

CONSTRAINTS:
--------------------------------------------------
Row 1: -x1 <= 0
Row 2: -x2 <= 0
Row 3: x2 <= 1
Row 4: -x3 <= 0
--------------------------------------------------

AS STRINGS:
--------------------------------------------------
Row 1: -frozenset({('demes', 0, 'epochs', 0, 'end_size'), ('demes', 0, 'epochs', 0, 'start_size')}) <= 0.0
Row 2: -frozenset({('migrations', 0, 'rate')}) <= 0.0
Row 3: frozenset({('migrations', 0, 'rate')}) <= 1.0
Row 4: -frozenset({('demes', 1, 'start_time'), ('migrations', 0, 'start_time'), ('migrations', 1, 'start_time'), ('demes', 2, 'start_time'), ('demes', 0, 'epochs', 0, 'end_time')}) <= 0.0
--------------------------------------------------
```

## Alternative representation of inequality constraints

Depending on the numerical optimizer one would like to use, sometimes it's preferable to express inequality constraints explicitly with a lower and upper bound using the ``alternative_constraint_rep`` function.

```python
from demestats.fit.util import alternative_constraint_rep

G, h = constraint["ineq"]
A_alt, ub_alt, lb_alt = alternative_constraint_rep(G, h)
print(A_alt)
print("lower bound: ", lb_alt)
print("upper bound: ", ub_alt)
```

The output:

```python
[[1. 0. 0.]
[0. 1. 0.]
[0. 0. 1.]]
lower bound: [0. 0. 0.]
upper bound: [inf  1. inf]
```

Which can be interpreted as:

```python
Row 1: 0 <= x1 <= inf
Row 2: 0 <= x2 <= 1
Row 3: 0 <= x3 <= inf
```

In the [`SFS Optimization`](https://demestats.readthedocs.io/en/latest/sfs_optimization.html) documentation, we will see that `scipy.minimize.LinearConstraint` requires this alternative representation of inequality constraints.

## Modifying the constraints

In addition to the constraints automatically derived from the construction of the demographic model, users may impose custom constraints to reflect specific biological assumptions or modeling choices. 

A common example is the symmetry constraint on migration rates. Using the same IWM model, let's work with 3 parameters - the ancestral population size and the symmetric migration rate between P0 and P1. We start by obtaining the default constraints:

```python
momi3_parameters = [
    frozenset({
        ("demes", 0, "epochs", 0, "end_size"),
        ("demes", 0, "epochs", 0, "start_size"),
    }), # Ancestral population size (index 0)
    frozenset({('migrations', 0, 'rate')}), # Rate of migration P0 to P1 (array index 1)
    frozenset({('migrations', 1, 'rate')}), # Rate of migration P1 to P0 (array index 2)
]

constraint = constraints_for(et, *momi3_parameters)
print(constraint)
```

There are currently no equality constraints. We can modify the constraint to enforce symmetry in migration rates using the ``modify_constraints_for_equality`` function. We provide the original ``constraint`` and an array of tuple ``indices`` for the variables that we want to apply an equality constraint.

```python
from demesinfer.fit.util import modify_constraints_for_equality

# variables in index positions 1 and 2 will be constrained to be equal
indices = [(1,2)]
new_constraint = modify_constraints_for_equality(constraint, indices)
print(new_constraint)
```

Sure enough, the updated constraint now includes the symmetry condition:

```python
{'eq': (array([[ 0.,  1., -1.]]), array([0.])), 
'ineq': (array([[-1., -0., -0.],
                [-0., -1., -0.],
                [ 0.,  1.,  0.],
                [-0., -0., -1.],
                [ 0.,  0.,  1.]]), 
        array([0., 0., 1., 0., 1.]))}
```

To get 3 or more variables to have an equality constraint, one must create separate constraints (e.g. indices = [(1,2), (2,3)] represent x1 = x2 and x2 = x3 which is equivalent to x1 = x2 = x3).

To modify the inequalities, one can directly modify the ``constraints`` object. For example, if we want to add the constraint that the ancestral population size >= 2000, we must change the constraints in the tuple (A_ineq, b_ineq). Note that we need to pay attention to negative signs and do the following:

```python
new_constraint["ineq"][1][0] = -2000.
```