from typing import NamedTuple, Self

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Scalar


class _base(eqx.Module):
    def partition(self):
        return eqx.partition(self, eqx.is_array_like)


class SetupState(_base):
    n: int = eqx.field(static=True)
    pops: tuple[str, ...] = eqx.field(static=True)

    @property
    def d(self) -> int:
        return len(self.pops)


class State(_base):
    log_s: Scalar
