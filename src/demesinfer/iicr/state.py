from typing import NamedTuple

import equinox as eqx
import jax
from jaxtyping import Array, Float, Scalar
from penzai import pz


class State(eqx.Module):
    p: pz.nx.NamedArray | jax.Array
    log_s: Scalar

    def __init__(self, p: pz.nx.NamedArray | Array, log_s: Scalar):
        self.p = p
        self.log_s = log_s

    def _replace(self, **kwargs):
        keys = list(kwargs.keys())
        vals = list(kwargs.values())
        where_fn = lambda tree: [getattr(tree, k) for k in keys]
        return eqx.tree_at(where_fn, self, vals)


class StateDn(State):
    # p is an (d,)*n  array denoting the joint probability that each of N lineages is
    # in each of D demes.
    # c is the probability that the first coalescence event has not occured by time t.
    pops: tuple[str, ...]

    def __init__(self, p: Array, log_s: Scalar, pops: tuple[str, ...]):
        self.p = p
        self.log_s = log_s
        self.pops = pops
        self.check_shape()

    @property
    def n(self) -> int:
        return self.p.ndim

    @property
    def d(self) -> int:
        return len(self.pops)

    def check_shape(self):
        assert self.p.shape == (self.d,) * self.n


class StateNd(State):
    # p is an (n+1,)*d  array denoting the joint probability that each of N lineages is
    # in each of D demes.
    # c is the probability that the first coalescence event has not occured by time t.
    @property
    def pops(self) -> tuple[str, ...]:
        return tuple(self.p.named_axes.keys())

    @property
    def d(self) -> int:
        return len(self.pops)

    @property
    def n(self) -> int:
        ret = next(iter(self.p.named_axes.values()))
        assert all(v == ret for v in self.p.named_axes.values()), (
            f"Expected all axes to have the same size, got {self.p.named_axes}"
        )
        return ret - 1


SetupState = None
StateReturn = tuple[State, dict]
SetupReturn = tuple[SetupState, dict]
