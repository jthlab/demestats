import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax import ShapeDtypeStruct, custom_vjp


@custom_vjp
def lyapunov(A: jnp.ndarray, Q: jnp.ndarray) -> jnp.ndarray:
    """Solve AX + X Aᵀ = Q via SciPy, JIT-safe with custom VJP.

    Args:
      A: (..., n, n)
      Q: (..., n, n)

    Returns:
      X: (..., n, n) satisfying A X + X Aᵀ = Q.
    """
    out_spec = ShapeDtypeStruct(shape=Q.shape, dtype=jnp.result_type(A, Q))
    return jax.pure_callback(scipy.linalg.solve_continuous_lyapunov, out_spec, A, Q)


def _lyapunov_fwd(
    A: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    X = lyapunov.fun(A, Q)
    return X, (A, X)


def _lyapunov_bwd(res, dX: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    A, X = res
    # Adjoint: solve Aᵀ Y + Y A = dX
    Y = lyapunov.fun(A.T, dX)
    dQ = Y
    # From ⟨dL/dX, dX⟩ = ⟨Y, dQ - dA X - X dAᵀ⟩ ⇒ dĀ = -(X Yᵀ + Xᵀ Y)
    dA = -(Y @ X.T + Y.T @ X)
    return dA, dQ


lyapunov.defvjp(_lyapunov_fwd, _lyapunov_bwd)

import jax.test_util


def test_lyapunov():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.standard_normal((n, n))
    Q = rng.standard_normal((n, n))
    X = lyapunov(A, Q)
    np.testing.assert_allclose(A @ X + X @ A.T, Q, atol=1e-5)


def test_lyapunov_grad():
    rng = np.random.default_rng(0)
    n = 5
    A = rng.standard_normal((n, n))
    Q = rng.standard_normal((n, n))

    def f(A, Q):
        X = lyapunov(A, Q)
        return X.sum()

    jax.test_util.check_grads(f, (A, Q), modes="rev", order=1)
