from numbers import Number
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from interpax import PPoly
from jax import vmap
from jaxtyping import Array, ArrayLike, Float, Scalar, ScalarLike


class PExp(NamedTuple):
    """Piecewise exponential rate function.

    This represents the function

        eta(t) = N0[i] (N1[i] / N0[i])^[(t-t[i])/(t[i+1]-t[i])] for t_i <= t < t_{i+1}

    i.e. eta(t[i])=N0[i], eta(t[i+1])=N1[i], and eta(t) is exponential between t[i] and t[i+1].

    The last time period is constrained to be constant.

    Args:
        N0, N1: positive arrays of shape [T] corresponding to the formula shown above.
        t: positive array of shape [T + 1] corresponding to t_i in the formula shown above.
    """

    N0: Float[ArrayLike, "T"]
    N1: Float[ArrayLike, "T"]
    t: Float[ArrayLike, "T+1"]

    @property
    def a(self):
        "eta(t) = a[i] exp(-(t[i + 1] - t)) b[i]) = 1 / (2 Ne(t))"
        return 1 / 2 / self.N1

    @property
    def b(self):
        "eta(t) = a[i] exp(-(t[i + 1]-t) b[i]) = 1 / (2 Ne(t))"
        # eta(t[i]) = a[i] exp(-b[i] dt[i]) = 1 / 2 / self.N0 =>
        return -jnp.log(1 / 2 / self.N0 / self.a) / jnp.diff(self.t)

    def __call__(self, u: ScalarLike) -> Scalar:
        r"Evaluate eta(u)."
        # log eta(t) = log(N1[i]) + [(t[i+1] - u)/(t[i+1]-t[i])] log(N0[i] / N1[i]) ^for t_i <= t < t_{i+1}
        # = log(N1[i]) + [1 - (u - t[i])/(t[i+1]-t[i])] log(N0[i] / N1[i]) ^for t_i <= t < t_{i+1}
        # = log(N0[i]) - (u - t[i])/(t[i+1]-t[i])] log(N0[i] / N1[i]) for t_i <= t < t_{i+1}

        # the last entry of t might be +inf, like if the deme has a start_time of +inf
        # in this case we must assume that N0=N1
        t = jnp.where(
            jnp.isinf(self.t[-1]), jnp.append(self.t[:-1], 2 * self.t[-2]), self.t
        )

        log_N0 = jnp.log(self.N0)
        log_N1 = jnp.log(self.N1)
        dt = jnp.diff(self.t)
        dt0 = jnp.isclose(dt, 0.0)
        dt_safe = jnp.where(dt0, 1.0, dt)
        c = jnp.array(
            [
                jnp.where(dt0, 0.0, (log_N1 - log_N0) / dt_safe),
                log_N0,
            ]
        )
        log_eta = PPoly(c=c, x=self.t, check=False, extrapolate=True)
        return jnp.exp(log_eta(u))

    def R(self, u: ScalarLike) -> Scalar:
        r"Evaluate R(u) = \int_t[0]^u eta(s) ds"
        a = self.a
        b = self.b
        t = self.t
        dt = jnp.diff(jnp.minimum(u, t))
        ui = jnp.where(u < t[:-1], t[:-1], jnp.where(t[1:] < u, t[1:], u))
        const = jnp.isclose(self.N0, self.N1)
        b_safe = jnp.where(const, 1.0, b)

        t1_safe = jnp.where(const, 1.0, t[1:])
        ui_safe = jnp.where(const, 1.0, ui)
        dt_safe = jnp.where(const, 1.0, dt)
        integrals = (
            # a / b_safe * jnp.exp(-b_safe * (t[1:] - ui)) * -jnp.expm1(-b_safe * dt)
            a
            / b_safe
            * jnp.exp(-b_safe * (t1_safe - ui_safe))
            * -jnp.expm1(-b_safe * dt_safe)
        )
        integrals = jnp.where(const, a * dt, integrals)
        return integrals.sum()

    def exp_integral(
        self, t0: ScalarLike, t1: ScalarLike, c: ScalarLike = 1.0
    ) -> Scalar:
        r"""Compute the integral $\int_t0^t1 exp[-c * (R(t) - R(t0))] dt$ for $R(t) = \int_0^s eta(s) ds$.

        Args:
            c: The constant multiplier of R(t) in the integral.
        Returns:
            The value of the integral.
        """
        Rt0 = self.R(t0)

        def f(N0i, N1i, ti, ti1):
            # \int_ti^ti1 exp(-c R(t)) dt
            # = \int_ti^ti1 exp(-c R(ti) - c \int_ti^t eta(s) ds) dt
            # = \int_ti^ti1 exp(-c R(ti) - c \int_ti^t (1/2N0) ds) dt, if N0=N1
            # = exp(-c R(ti)) \int_ti^ti1 exp(-c (t - ti) (1/2N0) ds) dt
            # = exp(-c R(ti)) (N0/c) -expm1(-c / N0) dt)
            ti1_safe = jnp.where(jnp.isinf(ti1), 2 * ti, ti1)
            i1 = (
                jnp.exp(-c * (self.R(ti) - Rt0))
                * (2 * N0i / c)
                * jnp.where(
                    jnp.isinf(ti1), 1.0, -jnp.expm1(-c / (2 * N0i) * (ti1_safe - ti))
                )
            )
            x1 = jnp.linspace(ti, ti1_safe, 1000)
            x2 = jnp.linspace(x1[1], x1[-1], 1000)
            x = jnp.sort(jnp.concatenate([x1, x2]))
            i2 = jnp.trapezoid(jnp.exp(-c * (vmap(self.R)(x) - Rt0)), x)
            # ti1 might be +inf, but in that case we assume that N0i=N1i
            # (constant growth in last epoch)
            return jnp.where(jnp.isclose(N0i, N1i) | jnp.isinf(ti1), i1, i2)

        tm = self.t.clip(t0, t1)
        return vmap(f)(self.N0, self.N1, tm[:-1], tm[1:]).sum()


class PConst(NamedTuple):
    """Piecewise constant rate function."""

    N: Float[ArrayLike, "T"]
    t: Float[ArrayLike, "T+1"]

    @property
    def c(self):
        "eta(t) = c[i] for t[i] <= t < t[i+1]"
        return 1 / 2 / self.N

    @property
    def _ppoly(self):
        return PPoly(c=self.c[None], x=self.t, check=False, extrapolate=True)

    def __call__(self, u: ScalarLike) -> Scalar:
        r"Evaluate eta(u)."
        return self._ppoly(u)

    def R(self, u: ScalarLike) -> Scalar:
        r"Evaluate R(u) = \int_t[0]^u eta(s) ds"
        return self._ppoly.integrate(0.0, u)
