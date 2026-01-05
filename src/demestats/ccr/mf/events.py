from dataclasses import dataclass

import jax.numpy as jnp

from ...iicr import events as base
from .state import StateMf as State

NoOp = base.NoOp
Epoch = base.Epoch
State = State


@dataclass(kw_only=True)
class PopulationStart(base.PopulationStart):
    pass


@dataclass(kw_only=True)
class Split1(base.Split1):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        pops = list(child_state.pops)
        assert self.donor in pops
        assert self.recipient in pops
        donor = pops.index(self.donor)
        recip = pops.index(self.recipient)
        r = child_state.r.at[recip].add(child_state.r[donor])
        b = child_state.b.at[recip].add(child_state.b[donor])
        keep = jnp.array([i for i in range(len(pops)) if i != donor], dtype=jnp.int32)
        pops.pop(donor)
        return (
            State(pops=tuple(pops), r=r[keep], b=b[keep], log_s=child_state.log_s),
            {},
        )


@dataclass(kw_only=True)
class Split2(base.Split2):
    def __call__(
        self, demo: dict, aux: dict, donor_state: State, recipient_state: State
    ) -> tuple[State, dict]:
        # Combine into recipient block, then perform Split1 logic.
        pops = list(recipient_state.pops) + list(donor_state.pops)
        r = jnp.concatenate([recipient_state.r, donor_state.r])
        b = jnp.concatenate([recipient_state.b, donor_state.b])
        donor = pops.index(self.donor)
        recip = pops.index(self.recipient)
        r = r.at[recip].add(r[donor])
        b = b.at[recip].add(b[donor])
        keep = jnp.array([i for i in range(len(pops)) if i != donor], dtype=jnp.int32)
        pops.pop(donor)
        return (
            State(
                pops=tuple(pops),
                r=r[keep],
                b=b[keep],
                log_s=donor_state.log_s + recipient_state.log_s,
            ),
            {},
        )


@dataclass(kw_only=True)
class Merge(base.Merge):
    def __call__(
        self, demo: dict, aux: dict, pop1_state: State, pop2_state: State
    ) -> tuple[State, dict]:
        assert set(pop1_state.pops).isdisjoint(set(pop2_state.pops))
        pops = pop1_state.pops + pop2_state.pops
        r = jnp.concatenate([pop1_state.r, pop2_state.r])
        b = jnp.concatenate([pop1_state.b, pop2_state.b])
        return (
            State(pops=pops, r=r, b=b, log_s=pop1_state.log_s + pop2_state.log_s),
            {},
        )


@dataclass(kw_only=True)
class MigrationStart(base.MigrationStart):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        return child_state, {}


@dataclass(kw_only=True)
class MigrationEnd(base.MigrationEnd):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        return child_state, {}


@dataclass(kw_only=True)
class Pulse(base.Pulse):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        p = self.prop_fun(demo)
        pops = list(child_state.pops)
        src = pops.index(self.source)
        dst = pops.index(self.dest)

        r = child_state.r
        b = child_state.b
        moved_r = p * r[dst]
        moved_b = p * b[dst]
        r = r.at[dst].add(-moved_r).at[src].add(moved_r)
        b = b.at[dst].add(-moved_b).at[src].add(moved_b)
        return State(pops=child_state.pops, r=r, b=b, log_s=child_state.log_s), {}


@dataclass(kw_only=True)
class Admix(base.Admix):
    def __call__(self, demo: dict, aux: dict, child_state: State) -> tuple[State, dict]:
        p = self.prop_fun(demo)
        pops = list(child_state.pops)
        child = pops.index(self.child)

        # ensure parents exist; if not, append them.
        r = child_state.r
        b = child_state.b
        if self.parent1 in pops:
            p1 = pops.index(self.parent1)
        else:
            pops.append(self.parent1)
            r = jnp.concatenate([r, jnp.zeros((1,), r.dtype)])
            b = jnp.concatenate([b, jnp.zeros((1,), b.dtype)])
            p1 = len(pops) - 1
        if self.parent2 in pops:
            p2 = pops.index(self.parent2)
        else:
            pops.append(self.parent2)
            r = jnp.concatenate([r, jnp.zeros((1,), r.dtype)])
            b = jnp.concatenate([b, jnp.zeros((1,), b.dtype)])
            p2 = len(pops) - 1

        r_child = r[child]
        b_child = b[child]
        r = r.at[p1].add((1 - p) * r_child).at[p2].add(p * r_child)
        b = b.at[p1].add((1 - p) * b_child).at[p2].add(p * b_child)
        keep = jnp.array([i for i in range(len(pops)) if i != child], dtype=jnp.int32)
        pops.pop(child)
        return (
            State(pops=tuple(pops), r=r[keep], b=b[keep], log_s=child_state.log_s),
            {},
        )


State = State
