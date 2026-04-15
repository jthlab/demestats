from dataclasses import replace
from typing import TypeVar

from beartype.typing import Callable
from jaxtyping import Int, ScalarLike

from .state import State

S = TypeVar("S", bound=State)


def execute(
    state: S,
    t0i: Int[ScalarLike, ""],
    t1i: Int[ScalarLike, ""],
    terminal: bool,
    constant: bool,
    migrations: list[tuple[str, str]],
    aux: dict,
    demo: dict,
    PanmicticInterp: type,
    lift_expm: Callable,
    lift_ode: Callable,
) -> tuple[S, dict]:
    t0 = demo["_times"][t0i]
    t1 = demo["_times"][t1i]
    if terminal or (not migrations):
        interp = PanmicticInterp(
            state=state,
            t0=t0,
            t1=t1,
        )
        if not terminal:
            st1 = interp(t1, demo)
            state = replace(state, p=st1["p"], log_s=st1["log_s"])
        return state, dict(interp=interp)
    elif constant:
        assert bool(migrations)
        return lift_expm(state=state, t0=t0, t1=t1, demo=demo, aux=aux)
    else:
        assert (not constant) and bool(migrations)
        return lift_ode(state=state, t0=t0, t1=t1, demo=demo, aux=aux)
