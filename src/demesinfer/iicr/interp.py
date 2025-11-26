from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike
from penzai import pz
from plum import Kind, dispatch

from .. import util
from .state import State


class Interpolator(eqx.Module, ABC):
    t0: ScalarLike
    t1: ScalarLike
    state: State

    @abstractmethod
    def jumps(self, demo): ...

    @abstractmethod
    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]: ...


class PanmicticInterp(Interpolator):
    def jumps(self, demo):
        etas = util.coalescent_rates(demo)
        ts = jnp.concatenate([etas[p].t for p in self.state.pops])
        ts = jnp.clip(ts, self.t0, self.t1)
        return jnp.sort(ts)


class MigrationInterp(Interpolator):
    state: State

    @abstractmethod
    def evaluate(self, t, demo): ...

    def __call__(self, t, demo):
        t = jnp.clip(t, self.t0, self.t1)
        p, s = self.evaluate(t)
        p /= _sum_array(p)
        R = self.state.coal_rate(t, demo)
        c = _sum_array(p * R)
        return dict(c=c, log_s=self.state.log_s + jnp.log1p(-s), p=p)


@dispatch
def _sum_array(arr: pz.nx.NamedArray) -> ScalarLike:
    return arr.unwrap(*arr.named_axes.keys()).sum()


@dispatch
def _sum_array(arr: Float[Array, "..."]) -> ScalarLike:
    return arr.sum()


class DfxInterp(MigrationInterp):
    jump_ts: Float[Array, "..."]
    sol: dfx.Solution

    def evaluate(self, t):
        return self.sol.evaluate(t)

    def jumps(self, demo):
        return self.jump_ts


class MergedInterp(eqx.Module):
    interps: list[Interpolator]

    def jumps(self, demo):
        r1 = jnp.array([f.t0 for f in self.interps])
        r2 = jnp.array([f.t1 for f in self.interps])
        r3 = jnp.concatenate([f.jumps(demo) for f in self.interps])
        return jnp.sort(jnp.concatenate([r1, r2, r3]))

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        c = log_s = 0.0

        for f in self.interps:
            mask = (f.t0 <= t) & (t < f.t1)
            y = f(t.clip(f.t0, f.t1), demo)
            c += jnp.where(mask, y["c"], 0.0)
            log_s += jnp.where(mask, y["log_s"], 0.0)

        return dict(c=c, log_s=log_s)
