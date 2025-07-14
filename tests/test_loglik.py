import jax.numpy as jnp
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
