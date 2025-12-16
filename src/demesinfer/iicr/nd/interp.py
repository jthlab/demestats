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
        p = pz.nx.nmap(interp1d)(
            t, self.ts.untag("t"), self.ps.untag("t"), method="linear", extrap=True
        )
        s = p.unwrap(*self.state.pops).sum()
        return p, 1 - s

    def jumps(self, demo):
        return jnp.array([])


class PanmicticNdInterp(interp.PanmicticInterp):
    def __call__(self, t, demo):
        etas = util.coalescent_rates(demo)
        pops = self.state.pops
        C = self.state.C
        t = jnp.asarray(t)

        R_vals = jnp.array([etas[pop].R(t) - etas[pop].R(self.t0) for pop in pops])
        Ne = jnp.array([etas[pop](t) for pop in pops])
        coal = 1 / 2 / Ne

        C_arr = C.unwrap(*pops, "n")
        broadcast_shape = (1,) * (C_arr.ndim - 1) + (len(pops),)
        R_broadcast = jnp.broadcast_to(R_vals.reshape(broadcast_shape), C_arr.shape)
        coal_broadcast = jnp.broadcast_to(coal.reshape(broadcast_shape), C_arr.shape)
        zero_mask = jnp.isclose(C_arr, 0.0)
        # Avoid ever evaluating 0 * inf by masking the RHS first.
        CR_arr = (C_arr * jnp.where(zero_mask, 0.0, R_broadcast)).sum(axis=-1)
        coal_weights = (C_arr * jnp.where(zero_mask, 0.0, coal_broadcast)).sum(axis=-1)
        CR = pz.nx.wrap(CR_arr, *pops)

        # prevent nan bugs in backward pass
        @pz.nx.nmap
        def g(p, cr):
            p0 = jnp.isclose(p, 0.0)
            psafe = jnp.where(p0, 1.0, p)
            log_p_prime = jnp.log(psafe) - cr
            log_p_prime = jnp.where(p0, -jnp.inf, log_p_prime)
            return log_p_prime.squeeze()

        log_p_prime = g(self.state.p, CR)
        lp = log_p_prime.unwrap(*pops)
        log_s_prime = jax.scipy.special.logsumexp(lp)
        all_neg_inf = jnp.isneginf(log_s_prime)
        log_s_safe = jnp.where(all_neg_inf, 0.0, log_s_prime)
        prob = jnp.exp(lp - log_s_safe)
        prob = jnp.where(all_neg_inf, jnp.zeros_like(prob), prob)
        p_prime = pz.nx.wrap(prob, *pops)
        c = jnp.where(
            all_neg_inf,
            0.0,
            (p_prime.unwrap(*pops) * coal_weights).sum(),
        )
        return dict(c=c, log_s=self.state.log_s + log_s_prime, p=p_prime)
