import diffrax as dfx
import equinox as eqx
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
    def __call__(self, t, demo):
        etas = util.coalescent_rates(demo)
        R = []
        for p in self.state.pops:
            R.append(etas[p].R(t) - etas[p].R(self.t0))
        # no coalescence allowed in the "untracked" deme
        R.append(0.0)
        R = jnp.array(R)
        log_p_prime = jnp.log(self.state.p) - self.C.dot(R)
        log_s_prime = jax.scipy.special.logsumexp(log_p_prime)
        p_prime = jnp.exp(log_p_prime - log_s_prime)
        coal = jnp.array([1 / 2 / etas[p](t) for p in self.state.pops])
        coal = jnp.append(coal, 0.0)
        # probability distribution conditional on no coalescence
        c = jnp.sum(p_prime * self.C.dot(coal))
        return dict(c=c, log_s=self.state.log_s + log_s_prime, p=p_prime)


class ODEInterpolator(_Interpolator):
    sol: dfx.Solution

    def _rate(self, t, demo):
        etas = util.coalescent_rates(demo)
        eta = jnp.array([1 / 2 / etas[pop](t) for pop in self.state.pops])
        eta = jnp.append(eta, 0.0)
        return self.C.dot(eta)

    def __call__(self, t, demo):
        y = self.sol.evaluate(t)
        p, s = y
        p /= p.sum()
        c = jnp.sum(p * self._rate(t, demo))
        return dict(c=c, log_s=self.state.log_s + jnp.log1p(-s), p=p)


class FilterInterp(eqx.Module):
    interps: list[_Interpolator]

    def __call__(self, t: ScalarLike, demo: dict) -> dict[str, ScalarLike]:
        c = log_s = 0.0
        for f in self.interps:
            y = f(t, demo)
            mask = (f.t0 <= t) & (t < f.t1)
            c += y["c"] * mask
            log_s += y["log_s"] * mask
        return dict(c=c, log_s=log_s)
