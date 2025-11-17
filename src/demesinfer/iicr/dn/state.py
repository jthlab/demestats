import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from demesinfer import util

from ..state import SetupState, State


class StateDn(State):
    # p is an (d,)*n  array denoting the joint probability that each of N lineages is
    # in each of D demes.
    # c is the probability that the first coalescence event has not occured by time t.
    p: jax.Array
    pops: tuple[str, ...] = eqx.field(static=True)

    @property
    def n(self) -> int:
        return self.p.ndim

    @property
    def d(self) -> int:
        return len(self.pops)

    @property
    def C(self) -> jax.Array:
        n = self.n
        d1 = self.d + 1
        inds = np.transpose(np.unravel_index(np.arange(d1**n), (d1,) * n))
        counts = jax.vmap(lambda b: jnp.bincount(b, length=d1))(inds).reshape(
            (d1,) * n + (d1,)
        )
        return counts * (counts - 1) / 2

    def coal_rate(self, t, demo):
        etas = util.coalescent_rates(demo)
        # there are two cases to consider
        eta = jnp.array([1 / 2 / etas[pop](t) for pop in self.pops])
        eta = jnp.append(eta, 0.0)
        # works for both regular and named arrays
        return self.C.dot(eta)

    def check_shape(self):
        assert self.p.shape == (self.d,) * self.n

    @classmethod
    def setup(cls, num_samples: dict[str, int], k: int) -> dict[str, "StateDn"]:
        # idea here is that the state is a (d+1, d+1, ..., d+1)-tensor where
        # T[i, j, ..., k] is the probability that lineage 1 is in deme i, lineage 2 is in deme j, etc.
        # deme d+1 is a special deme that represents the "outside" deme
        ret = {}
        i = -1
        for pop in num_samples:
            e = jnp.ones(k, dtype=jnp.int32)
            for pop1 in num_samples:
                for j in range(k):
                    accept = (pop1 == pop) & (j < num_samples[pop1])
                    i = jnp.where(accept, i + 1, i)
                    e1 = e.at[i].set(0)
                    e = jnp.where(accept, e1, e)
            p = jnp.zeros((2,) * k).at[tuple(e)].set(1.0)
            ret[pop] = cls(p=p, log_s=jnp.array(0.0), pops=(pop,))
        return ret
