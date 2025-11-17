from collections import OrderedDict
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar, ScalarLike
from penzai import pz


class SetupState(eqx.Module):
    migrations: frozenset[str]
    axes: OrderedDict[str, Int[ScalarLike, ""]]
    ns: dict[str, dict[str, Int[ScalarLike, ""]]]
    aux: dict | None = None

    def partition(self) -> tuple["SetupState", "SetupState"]:
        return eqx.partition(self, eqx.is_array_like)

    def _replace(self, **changes) -> "SetupState":
        for k in ["migrations", "axes", "ns", "aux"]:
            changes.setdefault(k, getattr(self, k))
        return SetupState(**changes)


class State(eqx.Module):
    """The state of a node in the event tree:

    Attributes:
        pl: the likelihood of the subtended leaf alleles conditional on the number of derived alleles at this node
        phi: the total expected branch length subtending the leaf alleles
        l0: do the leaves beneath this pl all have zero derived alleles?
    """

    pl: pz.nx.NamedArray
    phi: ScalarLike
    l0: ScalarLike

    def partition(self):
        return eqx.partition(self, eqx.is_array_like)

    def _replace(self, **changes) -> "State":
        for k in ["pl", "phi", "l0"]:
            changes.setdefault(k, getattr(self, k))
        return State(**changes)


StateReturn = tuple[State, dict]
SetupReturn = tuple[SetupState, dict]

__all__ = [
    "State",
    "SetupState",
    "StateReturn",
    "SetupReturn",
]
