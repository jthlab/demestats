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
    return jax.pure_callback(
        scipy.linalg.solve_continuous_lyapunov,
        out_spec,
        A,
        Q,
        vmap_method="expand_dims",
    )


def _lyapunov_fwd(
    A: jnp.ndarray, Q: jnp.ndarray
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray]]:
    X = lyapunov.fun(A, Q)
    return X, (A, X)


def _lyapunov_bwd(res, dX: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    A, X = res
    # Adjoint: solve Aᵀ Y + Y A = dX
    dQ = lyapunov.fun(A.T, dX)
    # From ⟨dL/dX, dX⟩ = ⟨Y, dQ - dA X - X dAᵀ⟩ ⇒ dĀ = -(X Yᵀ + Xᵀ Y)
    dA = -(dQ @ X.T + dQ.T @ X)
    return dA, dQ


lyapunov.defvjp(_lyapunov_fwd, _lyapunov_bwd)


@custom_vjp
def sylvester(A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray) -> jnp.ndarray:
    """Solve the Sylvester equation A X + X B = C via SciPy, but usable in JAX.

    Args:
      A: (..., m, m) matrix (no batching in this minimal version; use vmap externally).
      B: (..., n, n) matrix.
      C: (..., m, n) matrix.

    Returns:
      X: (..., m, n) matrix solving A X + X B = C.

    Notes:
      - This wrapper calls SciPy at runtime; it’s non-portable across backends, but
        works under JIT on CPU/host via pure_callback.
      - Gradients are implemented analytically via a custom VJP:
        If G is the cotangent for X, solve Aᵀ Y + Y Bᵀ = G, then
          dL/dC = Y
          dL/dA = X Yᵀ
          dL/dB = Yᵀ X
    """
    m = A.shape[-1]
    n = B.shape[-1]
    assert A.shape[-2:] == (m, m)
    assert B.shape[-2:] == (n, n)
    assert C.shape[-2:] == (m, n)

    out_spec = ShapeDtypeStruct(shape=C.shape, dtype=jnp.result_type(A, B, C))
    return jax.pure_callback(
        scipy.linalg.solve_sylvester, out_spec, A, B, C, vmap_method="expand_dims"
    )


def _sylvester_fwd(
    A: jnp.ndarray, B: jnp.ndarray, C: jnp.ndarray
) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    X = sylvester.fun(A, B, C)  # call the primal once
    # Save A, B, X for the backward
    return X, (A, B, X)


def _sylvester_bwd(
    res, dX: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    A, B, X = res
    # Adjoint equation: Aᵀ Y + Y Bᵀ = dX
    dC = sylvester.fun(A.T, B.T, dX)
    dA = -dC @ X.T
    dB = -X.T @ dC
    return dA, dB, dC


sylvester.defvjp(_sylvester_fwd, _sylvester_bwd)
