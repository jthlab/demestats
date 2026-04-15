# ICR Optimization

This page demonstrates a constrained optimization workflow with ``demestats``. It stands on its own, separate from the ICR tutorial.

This tutorial shows how to create a customizable SciPy optimizer for any demographic model using ``scipy.minimize``. It is *not* a primer on numerical optimization. SciPy's API/behaviour can change across
versions — **We are not responsible for any updates/errors made to scipy.minimize.**

You can find the code in `docs/tutorial_code/icr_optimization.ipynb`.

## Imports

```python
from itertools import combinations
from typing import Any, List, Mapping, Set, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import vmap
from loguru import logger
from scipy.optimize import LinearConstraint, minimize
from demestats.fit.util import (
    _dict_to_vec,
    _vec_to_dict,
    _vec_to_dict_jax,
    create_inequalities,
    make_whitening_from_hessian,
    pullback_objective,
)
from demestats.loglik.icr_loglik import icr_loglik
from demestats.icr import ICRCurve
```

## Revisiting the IWM model

```python
demo = msp.Demography()
demo.add_population(initial_size=5000, name="anc")
demo.add_population(initial_size=5000, name="P0")
demo.add_population(initial_size=5000, name="P1")
demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
demo.add_population_split(time=1000, derived=["P0", "P1"], ancestral="anc")

sample_size = 30 # we simulate 30 diploids
samples = {"P0": sample_size, "P1": sample_size}
ts = msp.sim_mutations(
    msp.sim_ancestry(
        samples=samples, demography=demo,
        recombination_rate=1e-8, sequence_length=1e8, random_seed=12
    ),
    rate=1e-8, random_seed=13
)

data, cfg_list = get_tree_from_positions_data_efficient(ts, num_samples=300, k=10, option="random", gap=200000, seed = 8)
```

Note that the `data` obtained is a set of `k`-order first-coalescence times across trees that are separated by a distance of `gap`.
`cfg_list` is the associated sampling configuration. Please refer to `get_tree_from_positions_data_efficient` in the API.

## Inspecting and selecting parameters to optimize

Inspect the parameters you can work with:

```python
et = EventTree(demo.to_demes())
et.variables
```

Suppose now we wish to optimize the following parameters, their associated values will be the initial guesses in the optimization process. We collect those parameters into a dictionary:

```python
paths = {frozenset({('demes', 0, 'epochs', 0, 'end_size'),
        ('demes', 0, 'epochs', 0, 'start_size')}):3000.,
        frozenset({('demes', 1, 'epochs', 0, 'end_size'),
            ('demes', 1, 'epochs', 0, 'start_size')}): 6000.,
        frozenset({('demes', 2, 'epochs', 0, 'end_size'),
            ('demes', 2, 'epochs', 0, 'start_size')}): 4000.}

cons = create_constraints(demo.to_demes(), paths)
```

The ``create_constraints`` function is a helper function that takes in a dictionary of parameters and calls on ``constraints_for`` to output the associated constraints. For any contrained optimization method, one needs a set of parameters they wish to optimize, the constraints, a way to compute the expected ICR, and a way to compute the likelihood and its gradient. 

## Initial required setup

```python
###### Part 1 #####
path_order: List[Var] = list(paths) # convert parameters into list
x0 = jnp.array(_dict_to_vec(paths, path_order)) # convert initial values into a vector
lb = jnp.array([0, 0, 0])
ub = jnp.array([1e8, 1e8, 1e8])
cfg_mat, deme_names = process_data(cfg_list)
# Note that we are choosing k = 10, entries of cfg_list MUST add up to the k you select
icr = ICRCurve(demo=demo, k=10) 
icr_call = jax.jit(icr.__call__)

##### Part 2 ######
args_nonstatic = (path_order, data, cfg_mat)
args_static = (icr_call, deme_names)
L, LinvT = make_whitening_from_hessian(_compute_icr_likelihood, x0, args_nonstatic, args_static)
preconditioner_nonstatic = (x0, LinvT)
g = pullback_objective(_compute_icr_likelihood, args_static)
y0 = np.zeros_like(x0)

lb_tr = L.T @ (lb - x0)
ub_tr = L.T @ (ub - x0)
```

For ``scipy.minimize``, the setup requires three parts. Construction of likelihood functions are described at the end of the tutorial.

##### Part 1 

- While ``dictionary`` representations provide intuitive tracking of parameter states, most numerical optimizers require vectorized inputs. To address this, parameter dictionaries must be transformed into compatible vector formats. The function `process_data` takes a dictionary `cfg_list` and splits it into two jax compatible vectors.
- Based on our benchmarking with scipy.minimize, we recommend explicitly defining lower (lb) and upper (ub) bounds to constrain the search space, which significantly improves optimization performance and stability.

#### Part 2 

This setup is optional and will depend on the user's preference. In our experience with ``scipy.minimize``, due to the magnitude difference between parameters such as the migration rate and population sizes, the gradient with respect to each variable will cause parameter updates to be unstable. We implemented a preconditioning method that makes our problem more suitable for optimization. Instead of optimizing over the classical likelihood function ``_compute_icr_likelihood``, we transform the likelihood function and the bounds to instead optimize over a function ``g`` with better conditioning. For more information on [preconditioning](https://en.wikipedia.org/wiki/Preconditioner).

## Constraints and scipy.optimize.LinearConstraint

Use ``constraints_for`` to derive the linear constraints for your chosen
parameters. It returns a dict with:

- ``"eq"`` → ``(Aeq, beq)`` for equality constraints
- ``"ineq"`` → ``(G, h)`` for inequalities.

These map directly to SciPy's ``scipy.optimize.LinearConstraint``:

```python
linear_constraints: list[LinearConstraint] = []

Aeq, beq = cons["eq"]
A_tilde = Aeq @ LinvT
b_tilde = beq - Aeq@x0
if Aeq.size:
    linear_constraints.append(LinearConstraint(A_tilde, b_tilde, b_tilde))

G, h = cons["ineq"]
if G.size:
    linear_constraints.append(create_inequalities(G, h, LinvT, x0, size=len(paths)))
```

As explained in the [`Model Constraints`](model_constraints), one would use ``create_inequalities`` to modify the output of ``constraints_for`` into the appropriate scipy.optimize.LinearConstraint format. 

## Create and run the optimizer

The final step is use an optimizer. Here we use ``scipy.minimize`` with constrained optimizer ``"trust-constr"``. Please refer to [`scipy.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)

```python
gtol = 1e-5
xtol = 1e-5
maxiter = 1000
barrier_tol = 1e-5

res = minimize(
    fun=neg_loglik,
    x0=y0,
    jac=True,
    args = (g, preconditioner_nonstatic, args_nonstatic, lb_tr, ub_tr),
    method=method,
    constraints=linear_constraints,
    options={
        'gtol': gtol,
        'xtol': xtol, 
        'maxiter': maxiter,
        'barrier_tol': barrier_tol,
    }
)

x_opt = np.array(x0) + LinvT @ res.x
```

Due to preconditioning, we must transform the variable to inspect our final estimates:

```python
print("Optimal parameters: ", _vec_to_dict(jnp.asarray(res.x), path_order))
print("\nFinal likelihood evaluation: ", res.fun)
print("Optimal parameters as a vector: ", x_opt)
```

For the simulated example, the estimates are close to the true values:

```python
Optimal parameters:  
{frozenset({('demes', 0, 'epochs', 0, 'end_size'), ('demes', 0, 'epochs', 0, 'start_size')}): 5016.814453125, 

frozenset({('demes', 1, 'epochs', 0, 'start_size'), ('demes', 1, 'epochs', 0, 'end_size')}): 5238.4287109375, 

frozenset({('demes', 2, 'epochs', 0, 'start_size'), ('demes', 2, 'epochs', 0, 'end_size')}): 5025.2666015625}
    
Final negative log-likelihood evaluation:  430751.8125
Optimal parameters as a vector:  [5016.8145 5238.4287 5025.2666]
```

We have this full pipeline described above wrapped in a single ``fit_icr`` function for convenience. See the `API` documentation for available options and implementation details. The convenience of `demestats` is that each component of the optimization pipeline can be modified and operated on its own, but if one wants to use the ``fit_icr`` function:

```python
from demestats.fit.fit_icr import fit

optimal_params, final_loglikelihood, optimal_params_vector = fit(demo.to_demes(), paths, data, cfg_list, cons, lb, ub, k=10)
```

## Understanding the objective function

Below is just an example of the objective function we designed for ``scipy.minimize`` to give users a better understanding of creating an inference pipeline. 

```python
def _compute_icr_likelihood(vec, args_nonstatic, args_static):
    path_order, data, cfg_mat = args_nonstatic
    icr_call, deme_names = args_static
    params = _vec_to_dict_jax(vec, path_order)
    jax.debug.print("param: {vec}", vec=vec)
    batched_loglik = vmap(compute_loglik, in_axes=(0, 0, None, None, None))(
        data, cfg_mat, params, icr_call, deme_names
    )
    # for debugging
    # jax.debug.print("batched_loglik: {}", batched_loglik)
    loss = -jnp.sum(batched_loglik)
    jax.debug.print("Loss: {loss}", loss=loss)
    return loss


def neg_loglik(vec, g, preconditioner_nonstatic, args_nonstatic, lb, ub):
    if jnp.any(vec >= ub):
        return jnp.inf, jnp.full_like(vec, 1e10)

    if jnp.any(vec <= lb):
        return jnp.inf, jnp.full_like(vec, -1e10)

    return g(vec, preconditioner_nonstatic, args_nonstatic)
```

The objective function one would typically use for scipy.minimize would look something like ``neg_loglik``, where ``vec`` represents the parameter values, ``g`` computes the likelihood and gradient in a transformed space by indirectly calling ``_compute_icr_likelihood``, and the other variables are arguments necessary for computing the likelihood. To limit the parameter search space we define variables ``lb`` and ``ub``. The ``_compute_icr_likelihood`` function unpacks all of the arguments and evalutes the ICR at every provided k-order first-coalescence time.