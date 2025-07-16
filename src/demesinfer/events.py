from dataclasses import dataclass, field

from beartype.typing import Callable, TypeVar

T = TypeVar("T")


@dataclass(kw_only=True)
class Event:
    """
    Base class for all events.
    """

    def setup(self, demo, aux, **kw):
        """
        Setup the event with the given state.
        This method should be implemented in subclasses.
        """
        raise NotImplementedError("Setup method must be implemented in subclasses.")

    def __call__(self, demo, aux, **kw):
        """
        Call the event with the given state and additional keyword arguments.
        """
        raise NotImplementedError("Call method must be implemented in subclasses.")


class NoOp(Event):
    """
    No operation event, used as a placeholder.
    """

    def setup(self, demo, aux, child_state: T) -> tuple[T, dict]:
        """
        Setup the no-op event.
        """
        return child_state, {}

    def __call__(self, demo, aux, child_state: T) -> tuple[T, dict]:
        """
        Call the no-op event.
        """
        return child_state, {}


@dataclass(kw_only=True)
class PopulationStart(NoOp):
    pass


@dataclass(kw_only=True)
class Merge(Event):
    """
    Merge two populations.
    """

    pop1: str
    pop2: str


@dataclass(kw_only=True)
class MigrationStart(Event):
    """
    Event triggered when a migration starts.
    """

    source: str
    dest: str


@dataclass(kw_only=True)
class MigrationEnd(Event):
    """
    Event triggered when a migration starts.
    """

    source: str
    dest: str


@dataclass(kw_only=True)
class Pulse(Event):
    """
    Event triggered by a pulse.
    """

    source: str
    dest: str
    prop_fun: Callable[[dict], float] = field(repr=False)


@dataclass(kw_only=True)
class Admix(Event):
    """
    Event triggered by a pulse.
    """

    child: str
    parent1: str
    parent2: str
    prop_fun: Callable[[dict], float] = field(repr=False)


@dataclass(kw_only=True)
class Split1(Event):
    """
    Event triggered by a split when populations are in same block.
    """

    donor: str
    recipient: str


@dataclass(kw_only=True)
class Split2(Event):
    """
    Event triggered by a split when populations are in different blocks.
    """

    donor: str
    recipient: str
