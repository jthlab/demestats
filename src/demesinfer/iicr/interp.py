from functools import partial

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, ScalarLike
from penzai import pz

import demesinfer.util as util

from .state import State


class Interpolator(eqx.Module):
    state: State
    t0: ScalarLike
    t1: ScalarLike


class PanmicticInterpolator(Interpolator):
    C: pz.nx.NamedArray

    def jumps(self, demo):
        etas = util.coalescent_rates(demo)
        ts = jnp.concatenate([etas[p].t for p in self.state.pops])
        ts = jnp.clip(ts, self.t0, self.t1)
        return jnp.sort(ts)

    def __call__(self, t, demo):
        etas = util.coalescent_rates(demo)
        pops = self.state.pops

        def f(t):
            R = jnp.array([etas[pop].R(t) - etas[pop].R(self.t0) for pop in pops])
            R = pz.nx.wrap(R, "n")
            coal = jnp.array([1 / 2 / etas[pop](t) for pop in pops])

            # prevent nan bugs in backward pass
            @pz.nx.nmap
            def g(p, cr):
                p0 = jnp.isclose(p, 0.0)
                psafe = jnp.where(p0, 1.0, p)
                log_p_prime = jnp.log(p) - cr
                log_p_prime = jnp.where(p0, -jnp.inf, log_p_prime)
                return log_p_prime.squeeze()

            CR = self.C.untag("n").dot(R.untag("n"))
            log_p_prime = g(self.state.p, CR)
            log_s_prime = jax.scipy.special.logsumexp(log_p_prime.unwrap(*pops))
            # probability distribution conditional on no coalescence
            p_prime = pz.nx.nmap(jnp.exp)(log_p_prime - log_s_prime)
            c = (p_prime * self.C.untag("n").dot(coal)).unwrap(*pops).sum()
            return dict(c=c, log_s=self.state.log_s + log_s_prime, p=p_prime)

        tinf = jnp.isinf(t)
        tsafe = jnp.where(tinf, self.t0, t)
        ret = f(tsafe)

        return jax.tree.map(
            partial(jnp.where, tinf),
            dict(c=0.0, log_s=self.state.log_s, p=self.state.p),
            ret,
        )


class FilterInterp(eqx.Module):
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
