from functools import partial

import diffrax as dfx
import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike

import demesinfer.util as util

from .state import State


class _Interpolator(eqx.Module):
    state: State
    t0: ScalarLike
    t1: ScalarLike
    C: Float[Array, "..."]


class PanmicticInterpolator(_Interpolator):
    def jumps(self, demo):
        etas = util.coalescent_rates(demo)
        ts = jnp.concatenate([etas[p].t for p in self.state.pops])
        ts = jnp.clip(ts, self.t0, self.t1)
        return jnp.sort(ts)

    def __call__(self, t, demo):
        etas = util.coalescent_rates(demo)

        def f(t):
            R = []
            for p in self.state.pops:
                R.append(etas[p].R(t) - etas[p].R(self.t0))
            # no coalescence allowed in the "untracked" deme
            R.append(0.0)
            R = jnp.array(R)

            # prevent nan bugs in backward pass
            p0 = jnp.isclose(self.state.p, 0.0)
            psafe = jnp.where(p0, 1.0, self.state.p)
            log_p_prime = jnp.log(psafe) - self.C.dot(R)
            log_p_prime = jnp.where(p0, -jnp.inf, log_p_prime)

            log_s_prime = jax.scipy.special.logsumexp(log_p_prime)
            p_prime = jnp.exp(log_p_prime - log_s_prime)
            coal = jnp.array([1 / 2 / etas[p](t) for p in self.state.pops])
            coal = jnp.append(coal, 0.0)
            # probability distribution conditional on no coalescence
            c = jnp.sum(p_prime * self.C.dot(coal))
            return dict(c=c, log_s=self.state.log_s + log_s_prime, p=p_prime)

        tinf = jnp.isinf(t)
        tsafe = jnp.where(tinf, self.t0, t)
        ret = f(tsafe)
        return jax.tree.map(
            partial(jnp.where, tinf),
            dict(c=0.0, log_s=self.state.log_s, p=self.state.p),
            ret,
        )


class ODEInterpolator(_Interpolator):
    jump_ts: Float[Array, "..."]
    f: interpax.Interpolator1D

    def __init__(self, sol: dfx.Solution, jump_ts: Float[Array, "..."], **kwargs):
        super().__init__(**kwargs)
        self.jump_ts = jump_ts
        x = jnp.linspace(sol.t0, sol.t1, 100)
        y = jax.vmap(sol.evaluate)(x)
        fs = [interpax.Interpolator1D(x, yy, extrap=True) for yy in y]

        def f(t):
            return fs[0](t), fs[1](t)

        self.f = f

    def jumps(self, demo):
        return self.jump_ts

    def _rate(self, t, demo):
        etas = util.coalescent_rates(demo)
        eta = jnp.array([1 / 2 / etas[pop](t) for pop in self.state.pops])
        eta = jnp.append(eta, 0.0)
        return self.C.dot(eta)

    def __call__(self, t, demo):
        t = jnp.clip(t, self.t0, self.t1)
        y = self.f(t)
        p, s = y
        p /= p.sum()
        c = jnp.sum(p * self._rate(t, demo))
        return dict(c=c, log_s=self.state.log_s + jnp.log1p(-s), p=p)


class FilterInterp(eqx.Module):
    interps: list[_Interpolator]

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
