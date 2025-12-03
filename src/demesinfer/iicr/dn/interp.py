from functools import partial

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from interpax import interp1d
from jaxtyping import Array, Float, ScalarLike

import demesinfer.util as util

from .. import interp
from ..interp import DfxInterp
from .state import State


class PanmicticDnInterp(interp.PanmicticInterp):
    def __call__(self, t, demo):
        etas = util.coalescent_rates(demo)
        C = self.state.C

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
            log_p_prime = jnp.log(psafe) - C.dot(R)
            log_p_prime = jnp.where(p0, -jnp.inf, log_p_prime)

            log_s_prime = jax.scipy.special.logsumexp(log_p_prime)
            p_prime = jnp.exp(log_p_prime - log_s_prime)
            coal = jnp.array([1 / 2 / etas[p](t) for p in self.state.pops])
            coal = jnp.append(coal, 0.0)
            # probability distribution conditional on no coalescence
            c = jnp.einsum("...,...c,c->", p_prime, C, coal)
            return dict(c=c, log_s=self.state.log_s + log_s_prime, p=p_prime)

        tinf = jnp.isinf(t)
        tsafe = jnp.where(tinf, self.t0, t)
        ret = f(tsafe)
        return jax.tree.map(
            partial(jnp.where, tinf),
            dict(c=0.0, log_s=self.state.log_s, p=self.state.p),
            ret,
        )


class ExpmDnInterp(interp.MigrationInterp):
    ts: Float[Array, "t"]
    ps: Float[Array, "t ..."]

    def evaluate(self, t):
        p = interp1d(t, self.ts, self.ps, method="linear", extrap=True)
        s = p.sum()
        return p, 1 - s

    def jumps(self, demo):
        return jnp.array([])
