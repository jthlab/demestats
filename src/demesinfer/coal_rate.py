from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass

import interpax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Scalar, ScalarLike


@dataclass(kw_only=True)
class CoalRate(ABC):
    @abstractproperty
    def jumps(self) -> Float[Array, "_"]:
        """Return the times at which the coalescent rate changes discontinuously."""
        pass

    @abstractmethod
    def __call__(self, t: ScalarLike) -> Scalar:
        """Evaluate the coalescent rate at time t."""
        pass

    def R(self, a: ScalarLike, b: ScalarLike) -> Scalar:
        r"""Integrated coalescent rate,

        int_a^b self(t) dt
        """
        pass


@dataclass(kw_only=True)
class PiecewiseConstant(CoalRate):
    """Piecewise constant coalescent rate."""

    c: Float[ArrayLike, "T"]
    t: Float[ArrayLike, "T"]

    @property
    def jumps(self) -> Float[Array, "T"]:
        """Return the times at which the coalescent rate changes discontinuously."""
        return jnp.array(self.t)

    @property
    def _ppoly(self) -> interpax.PPoly:
        return interpax.PPoly(self.c[None], jnp.append(self.t, jnp.inf), check=False)

    def __call__(self, t: ScalarLike) -> Scalar:
        return self._ppoly(t)

    def R(self, a: ScalarLike, b: ScalarLike) -> Scalar:
        return self._ppoly.integrate(a, b)
