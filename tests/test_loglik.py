from functools import partial

import jax
import jax.numpy as jnp
import msprime as msp
from pytest import fixture

import demesinfer.coal_rate
import demesinfer.loglik
from demesinfer.coal_rate import PiecewiseConstant


@fixture
def eta():
    return PiecewiseConstant(c=jnp.array([1.0, 0.5]), t=jnp.array([0.0, 10.0]))


def test_loglik(eta):
    r = 0.01
    data = jnp.array([[5.0, 10.0], [8.0, 15.0], [12.0, 20.0]])

    # Call the loglik function
    result = demesinfer.loglik.loglik(eta, r, data)

    # Check if the result is a scalar
    assert isinstance(result, jnp.ndarray)
    assert result.ndim == 0

    # Check if the result is a valid number (not NaN or Inf)
    assert not jnp.isnan(result)
    assert not jnp.isinf(result)


def test_sample():
    sim = msp.simulate(
        2,
        mutation_rate=1e-8,
        recombination_rate=1e-8,
        Ne=10000,
        length=1e7,
        random_seed=42,
    )

    a = []
    for tree in sim.trees():
        t = tree.get_tmrca(0, 1)
        if a and t == a[-1][0]:
            a[-1][1] += tree.span
        else:
            a.append([tree.get_tmrca(0, 1), tree.span])
    data = jnp.array(a)

    @jax.jit
    @partial(jax.vmap, in_axes=(0, None))
    def f(N, data):
        eta = demesinfer.coal_rate.PiecewiseConstant(
            c=jnp.array([1 / 2 / N]), t=jnp.array([0.0])
        )
        r = 1e-8
        return demesinfer.loglik.loglik(eta, r, data)

    x = jnp.linspace(5_000, 50_000, 10)
    result = f(x, data)
