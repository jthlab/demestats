import jax
import jax.numpy as jnp
import jax.test_util
import numpy as np

from demesinfer.linalg import lyapunov, sylvester


def test_sylvester():
    rng = np.random.default_rng(1)
    A, B, C = rng.normal(size=(3, 8, 8))
    X = sylvester(A, B, C)
    np.testing.assert_allclose(A @ X + X @ B, C)


def test_sylvester_grad():
    rng = np.random.default_rng(1)
    A, B, C = rng.normal(size=(3, 8, 8))

    def loss_fn(A, B, C):
        X = sylvester(A, A.T, C)
        return X.sum()

    jax.test_util.check_grads(loss_fn, (A, B, C), modes="rev", order=1)


def test_lyapunov():
    rng = np.random.default_rng(0)
    n = 5
    A, Q = rng.normal(size=(2, n, n))
    X = lyapunov(A, Q)
    np.testing.assert_allclose(A @ X + X @ A.T, Q, atol=1e-5)


def test_lyapunov_grad():
    rng = np.random.default_rng(0)
    n = 5
    A, Q = rng.standard_normal(size=(2, n, n))

    def f(A, Q):
        X = lyapunov(A, Q)
        return X.sum()

    jax.test_util.check_grads(f, (A, Q), modes="rev", order=1)


def test_sql_eq_lya():
    rng = np.random.default_rng(0)
    n = 5
    A, Q = rng.standard_normal(size=(2, n, n))
    X1 = sylvester(A, A.T, Q)
    X2 = lyapunov(A, Q)
    np.testing.assert_allclose(X1, X2, atol=1e-5)


def test_sql_eq_lya_grad():
    rng = np.random.default_rng(0)
    n = 2
    A, Q = rng.standard_normal(size=(2, n, n))

    def f1(A, Q):
        X = sylvester(A, A.T, Q)
        return X.sum()

    def f2(A, Q):
        X = lyapunov(A, Q)
        return X.sum()

    g1 = jax.grad(f1)(A, Q)
    g2 = jax.grad(f2)(A, Q)
    np.testing.assert_allclose(g1, g2, atol=1e-5)
