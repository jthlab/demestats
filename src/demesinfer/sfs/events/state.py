from collections import OrderedDict
from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar, ScalarLike
from penzai import pz


class SetupState(NamedTuple):
    migrations: frozenset[str]
    axes: OrderedDict[str, int]
    ns: dict[str, dict[str, int]]
    aux: dict = None


class State(NamedTuple):
    """The state of a node in the event tree:

    Attributes:
        pl: the likelihood of the subtended leaf alleles conditional on the number of derived alleles at this node
        phi: the total expected branch length subtending the leaf alleles
        l0: do the leaves beneath this pl all have zero derived alleles?
    """

    pl: pz.nx.NamedArray
    phi: Scalar
    l0: Scalar


StateReturn = tuple[State, dict]
SetupReturn = tuple[SetupState, dict]

__all__ = [
    "State",
    "SetupState",
    "StateReturn",
    "SetupReturn",
]
