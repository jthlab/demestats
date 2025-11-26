from functools import partial

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from interpax import interp1d
from jaxtyping import Array, Float, ScalarLike
from penzai import pz

import demesinfer.util as util

from .. import interp
from ..interp import DfxInterp
from .state import State


class ExpmNdInterp(interp.MigrationInterp):
    ts: pz.nx.NamedArray
    ps: pz.nx.NamedArray

    def evaluate(self, t):
        p = pz.nx.nmap(interp1d)(t, self.ts.untag("t"), self.ps.untag("t"), extrap=True)
        s = p.unwrap(*self.state.pops).sum()
        return p, 1 - s

    def jumps(self, demo):
        return jnp.array([])


class PanmicticNdInterp(interp.PanmicticInterp):
    def __call__(self, t, demo):
        etas = util.coalescent_rates(demo)
        pops = self.state.pops
        C = self.state.C

        R = jnp.array([etas[pop].R(t) - etas[pop].R(self.t0) for pop in pops])
        coal = jnp.array([1 / 2 / etas[pop](t) for pop in pops])
        R = pz.nx.wrap(R, "n")

        # prevent nan bugs in backward pass
        @pz.nx.nmap
        def g(p, cr):
            p0 = jnp.isclose(p, 0.0)
            psafe = jnp.where(p0, 1.0, p)
            log_p_prime = jnp.log(psafe) - cr
            log_p_prime = jnp.where(p0, -jnp.inf, log_p_prime)
            return log_p_prime.squeeze()

        CR = C.untag("n").dot(R.untag("n"))
        log_p_prime = g(self.state.p, CR)
        log_s_prime = jax.scipy.special.logsumexp(log_p_prime.unwrap(*pops))
        # probability distribution conditional on no coalescence
        p_prime = pz.nx.nmap(jnp.exp)(log_p_prime - log_s_prime)
        c = (p_prime * C.untag("n").dot(coal)).unwrap(*pops).sum()
        return dict(c=c, log_s=self.state.log_s + log_s_prime, p=p_prime)
