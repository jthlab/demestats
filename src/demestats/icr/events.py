from dataclasses import dataclass

from .. import events as base
from .state import SetupState

NoOp = base.NoOp
Epoch = base.Epoch


@dataclass(kw_only=True)
class PopulationStart(base.PopulationStart):
    pass


@dataclass(kw_only=True)
class Split1(base.Split1):
    def setup(
        self,
        demo: dict,
        aux: dict,
        child_state: SetupState,
    ) -> tuple[SetupState, dict]:
        cp = list(child_state.pops)
        cp.remove(self.donor)
        return SetupState(pops=tuple(cp), n=child_state.n), {}


@dataclass(kw_only=True)
class Split2(base.Split2):
    def setup(
        self,
        demo: dict,
        aux: dict,
        donor_state: SetupState,
        recipient_state: SetupState,
    ) -> tuple[SetupState, dict]:
        combined_pops = recipient_state.pops + donor_state.pops
        assert self.donor in combined_pops
        cp = list(combined_pops)
        cp.remove(self.donor)
        return SetupState(pops=tuple(cp), n=donor_state.n), {}


@dataclass(kw_only=True)
class Merge(base.Merge):
    def setup(
        self, demo: dict, aux: dict, pop1_state: SetupState, pop2_state: SetupState
    ) -> tuple[SetupState, dict]:
        assert set.isdisjoint(set(pop1_state.pops), set(pop2_state.pops))
        cp = pop1_state.pops + pop2_state.pops
        return SetupState(pops=tuple(cp), n=pop1_state.n), {}


@dataclass(kw_only=True)
class MigrationStart(base.MigrationStart):
    def setup(
        self, demo: dict, aux: dict, child_state: SetupState
    ) -> tuple[SetupState, dict]:
        return child_state, {}


@dataclass(kw_only=True)
class MigrationEnd(base.MigrationEnd):
    def setup(
        self, demo: dict, aux: dict, child_state: SetupState
    ) -> tuple[SetupState, dict]:
        return child_state, {}


@dataclass(kw_only=True)
class Admix(base.Admix):
    def setup(
        self, demo: dict, aux: dict, child_state: SetupState
    ) -> tuple[SetupState, dict]:
        pops = set(child_state.pops) | {self.parent1, self.parent2}
        return SetupState(pops=tuple(pops), n=child_state.n), {}


@dataclass(kw_only=True)
class Pulse(base.Pulse):
    def setup(
        self, demo: dict, aux: dict, child_state: SetupState
    ) -> tuple[SetupState, dict]:
        return child_state, {}
