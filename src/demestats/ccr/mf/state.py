import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, ScalarLike

from ...iicr.state import State


class StateMf(State):
    pops: tuple[str, ...] = eqx.field(static=True)
    r: Float[Array, "d"]
    b: Float[Array, "d"]

    @property
    def d(self) -> int:
        return len(self.pops)

    @classmethod
    def setup(
        cls,
        num_samples: dict[str, tuple[Int[ScalarLike, ""], Int[ScalarLike, ""]]],
        k: int,
    ) -> dict[str, "StateMf"]:
        ret: dict[str, StateMf] = {}
        for pop, (nr, nb) in num_samples.items():
            nr = jnp.asarray(nr, dtype=jnp.float32)
            nb = jnp.asarray(nb, dtype=jnp.float32)
            ret[pop] = cls(
                pops=(pop,),
                r=nr.reshape((1,)),
                b=nb.reshape((1,)),
                log_s=jnp.array(0.0),
            )
        return ret
