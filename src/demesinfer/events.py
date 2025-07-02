from typing import Callable
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class Event:
    """
    Base class for all events.
    """

    pass


@dataclass(kw_only=True)
class PopulationStart(Event):
    pass


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
class Lift(Event):
    """
    Event triggered by a lift.
    """

    pass


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


@dataclass(kw_only=True)
class Rename(Event):
    """
    Event triggered by a rename.
    """

    old: str
    new: str
