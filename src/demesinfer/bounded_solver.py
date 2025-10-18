import diffrax as dfx
import jax
import jax.numpy as jnp
from beartype.typing import Callable
from diffrax._custom_types import VF, Args, BoolScalarLike, DenseInfo, RealScalarLike, Y
from diffrax._solver.base import _SolverState
from jaxtyping import PyTree


class BoundedSolver(dfx.Kvaerno3):
    oob_fn: Callable[[Y], BoolScalarLike] = None

    def step(
        self,
        terms: PyTree[dfx.AbstractTerm],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: _SolverState,
        made_jump: BoolScalarLike,
    ) -> tuple:
        y1, y_error, *rest = super().step(
            terms, t0, t1, y0, args, solver_state, made_jump
        )
        oob = self.oob_fn(y1)
        keep = lambda a: jnp.where(oob, jnp.inf, a)
        y_error = jax.tree.map(keep, y_error)
        return y1, y_error, *rest
