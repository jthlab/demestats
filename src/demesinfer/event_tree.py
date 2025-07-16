import math
from collections.abc import Collection
from enum import Enum
from functools import total_ordering
from itertools import count
from numbers import Number
from secrets import token_hex
from types import ModuleType

import demes
import networkx as nx
import numpy as np
from beartype import beartype
from beartype.typing import Callable, Iterable
from loguru import logger

import demesinfer.events
from demesinfer.events import Event

from .path import Path, get_path


@beartype
def unique_strs(col: Collection[str], k: int = 1, ell: int = 8) -> list[str]:
    "return k unique strings of length 2 * ell which are not in col"
    # each byte is converted to two hex digits, so the length of the string is 2 * ell
    ret = []
    while len(ret) < k:
        s = token_hex(ell)
        if s not in col and s not in ret:
            ret.append(s)
    return ret


@total_ordering
class EventType(Enum):
    MIGRATION_START = 1
    MIGRATION_END = 2
    EPOCH = 3
    PULSE = 4
    MERGE = 5
    POPULATION_START = 6

    def __lt__(self, other: "EventType") -> bool:
        assert isinstance(other, EventType)
        return self.value < other.value


def _all_events(demo: demes.Graph) -> Iterable[dict]:
    """Iterate over all events in the demes graph"""
    d = demo.asdict()
    for i, deme in enumerate(d["demes"]):
        name = deme["name"]
        if deme["ancestors"]:
            # merge events
            time_path = ("demes", i, "start_time")
            yield dict(
                t=time_path,
                pop=name,
                ancestors=deme["ancestors"],
                i=i,
                ev=EventType.MERGE,
            )
        else:
            # deme has no ancestors, so it must extend infinitely back into the past
            assert math.isinf(deme["start_time"])
            time_path = ("demes", i, "start_time")
            yield dict(
                t=time_path,
                pop=name,
                ev=EventType.POPULATION_START,
            )
    # pulse admixtures
    for j, p in enumerate(d["pulses"]):
        time_path = ("pulses", j, "time")
        yield dict(
            t=time_path,
            i=j,
            pop=p["dest"],
            sources=p["sources"],
            ev=EventType.PULSE,
        )
    # migration start
    for j, m in enumerate(d["migrations"]):
        rate_path = ("migrations", j, "rate")
        y = dict(i=j, source=m["source"], pop=m["dest"], rate=rate_path)
        # start and end are backwards for us since we are working in reverse time
        time_path = ("migrations", j, "end_time")
        yield y | dict(
            t=time_path,
            ev=EventType.MIGRATION_START,
        )
        time_path = ("migrations", j, "start_time")
        yield y | dict(
            t=time_path,
            ev=EventType.MIGRATION_END,
            rate=rate_path,
        )


Node = int


class EventTree:
    """Build an event tree from a demes graph.

    Args:
        demo: a demes graph
        events: a module containing event classes
    """

    def __init__(
        self,
        demo: demes.Graph,
        events: ModuleType = demesinfer.events,
    ):
        # store passed parameters
        self._demo = demo
        self._events = events

        # data structures for tree
        self._T = nx.DiGraph()
        self._i = count(0)

        # tree creation
        self._init_leaves()
        self._build_tree()
        self._check()
        self._check()

    @property
    def T(self):
        """Get the event tree."""
        return self._T

    @property
    def demo(self):
        """Get the demes graph."""
        return self._demo

    def demodict(self):
        """Get the demes graph as a dictionary."""
        return self._demo.asdict()

    @property
    def events(self):
        return self._events

    def leaves_below(self, i: Node) -> Collection[Node]:
        """Return all leaf nodes below node i."""
        return {n for n in nx.ancestors(self._T, i) if self._T.in_degree(n) == 0}

    @property
    def root(self):
        return next(n for n in self._T if self._T.out_degree(n) == 0)

    @property
    def leaves(self):
        return self.leaves_below(self.root)

    def _init_leaves(self):
        # Initialize leaf nodes for each population
        for j, deme in enumerate(self._demo.demes):
            # add initial leaf nodes for each population
            # attached to each node are attributes that track the population size and
            # migration rates. (these are the two model attributes that persist
            # over time).
            kw = dict(
                t=(
                    "demes",
                    j,
                    "epochs",
                    -1,
                    "end_time",
                ),  # time of the population start
                block=frozenset([deme.name]),
                event=self._events.PopulationStart(),
            )
            self._add_node(**kw)

    def _build_tree(self):
        """build the event tree"""
        events = self.events

        # this sorting function ensures that:
        # - events are processed (reverse-)chronologically
        # - contemporaneous events are processed in the order specified by EventType
        # - contemporaneous events of the same type are processed according to their
        #   order specified by demes.
        # the last point matters for simultaneous pulses in particular:
        # https://popsim-consortium.github.io/demes-spec-docs/main/specification.html#example-sequential-application-of-pulses  # noqa: E501
        def keyfun(d):
            return (self.get_path(d["t"]), d["ev"], d.get("i"))

        # iterate over all events in the sort order specified above
        for d in sorted(_all_events(self._demo), key=keyfun):
            t = d["t"]
            u = self._active(d["pop"])

            if d["ev"] == EventType.MIGRATION_START:
                v = self._active(d["source"])
                if self.nodes[u]["block"] != self.nodes[v]["block"]:
                    # these populations are all in the same block so we don't need to merge them
                    w = self._merge_nodes(
                        u, v, t=t, event=events.Merge(pop1=d["source"], pop2=d["pop"])
                    )
                    self.edges[u, w]["label"] = "pop1"
                    self.edges[v, w]["label"] = "pop2"
                    v = w
                assert d["source"] in self.nodes[v]["block"]
                assert d["pop"] in self.nodes[v]["block"]
                ev = events.MigrationStart(source=d["source"], dest=d["pop"])
                w = self._add_node(t=t, block=self.nodes[v]["block"], event=ev)
                self._add_edge(v, w)
                continue

            elif d["ev"] == EventType.MIGRATION_END:
                v = self._active(d["source"])
                assert u is v
                ev = events.MigrationEnd(source=d["source"], dest=d["pop"])
                w = self._add_node(t=t, block=self.nodes[u]["block"], event=ev)
                self._add_edge(u, w)
                continue

            # pulses function in a similarly to continuous migrations, but they are not
            # recorded in the state since they happen instantly.

            elif d["ev"] == EventType.PULSE:
                # From https://popsim-consortium.github.io/demes-spec-docs/main/specification.html#example-sequential-application-of-pulses  # noqa: E501
                # 1. Initialize an array of zeros with length equal to the num. demes.
                # 2. Set the ancestry proportion of the destination deme to 1.
                # 3. For each pulse:
                #    a. Multiply the array by one (1) minus the sum of proportions.
                #    b. For each source, add its proportion to the array.
                for j, s in enumerate(d["sources"]):
                    prop_path = ("pulses", d["i"], "proportions", j)

                    def pf(params: dict, prop_path=prop_path):
                        return get_path(params, prop_path)

                    self._pulse(source=s, dest=d["pop"], t=t, prop_fun=pf)

            elif d["ev"] == EventType.MERGE:
                # the population merges with ancestral population(s). we model this as a
                # sequence of pulses, followed by admixture.
                for j, s in enumerate(d["ancestors"][:-1]):

                    def pf(params, i=d["i"], j=j):
                        deme = params["demes"][i]
                        p = sum(deme["proportions"][:j])
                        # at the j-th pulse a fraction 1 - p of the population remains
                        # to be admixed
                        return deme["proportions"][j] / (1 - p)

                    # u is updated by the pulse
                    u = self._pulse(source=s, dest=d["pop"], t=t, prop_fun=pf)
                # the remaining ancestor merges with last ancestor
                s = d["ancestors"][-1]
                v = self._active(s)
                if d["pop"] in self.nodes[v]["block"]:
                    # the populations are already in the same block
                    w = self._add_node(
                        event=events.Split1(donor=d["pop"], recipient=s),
                        block=self.nodes[v]["block"] - {d["pop"]},
                        t=t,
                    )
                    self._add_edge(v, w)
                else:
                    ev = events.Split2(donor=d["pop"], recipient=s)
                    w = self._merge_nodes(u, v, t=t, event=ev, rm=d["pop"])
                    # identify which edge is which for later traversal
                    self.edges[u, w]["label"] = "donor"
                    self.edges[v, w]["label"] = "recipient"

            elif d["ev"] == EventType.POPULATION_START:
                # the population extends infinitely far back into the past. basically
                # just a lifting event.
                pass

            else:
                raise RuntimeError(f"unknown event type {d['ev']}")

        # now add a single edge extending infinitely back to the past
        r = self.root
        assert len(self.nodes[r]["block"]) == 1
        pop = list(self.nodes[r]["block"])[0]
        i = [d.name for d in self._demo.demes].index(pop)
        assert np.isinf(self._demo.demes[i].start_time)
        u = self._add_node(
            t=("demes", i, "start_time"),
            block=self.nodes[r]["block"],
            event=events.NoOp(),
        )
        self._add_edge(r, u)

        assert nx.is_tree(self._T)  # sanity check.

    def constraints(self, paths: Collection[Path] = None):
        if paths is None:
            paths = self._paths
        return self.reparameterize(paths).constraints

    @property
    def nodes(self):
        """Get the node with id i."""
        return self._T.nodes

    @property
    def edges(self):
        return self._T.edges

    def get_path(self, p: Path) -> Number:
        """Get the value of path p in the event tree."""
        return get_path(self.demodict(), p)

    @beartype
    def _time(self, node: Node) -> Number:
        ret = {self.get_path(p) for p in self.nodes[node]["t"]}
        assert len(ret) == 1, (node, self.nodes[node]["t"], ret)
        return ret.pop()

    def _check(self):
        assert nx.is_tree(self._T)  # sanity check.
        # check that the block each node comprises the populations beneath that node
        root = next(n for n in self._T if self._T.out_degree(n) == 0)
        assert len(self.nodes[root]["block"]) == 1

    @beartype
    def _add_edge(self, u: Node, v: Node):
        succ = list(self._T.successors(u))
        assert not succ, (u, v, succ)
        logger.debug("adding edge {} -> {}", u, v)
        self._T.add_edge(u, v)

    @beartype
    def _add_node(self, /, t: Path | float, block: frozenset[str], **kw) -> Node:
        """return a node which has the same blocks, (optionally) time, and attributes
        as u"""
        i = next(self._i)
        kw.update(
            {
                "block": block,
                "t": t,
            }
        )
        logger.debug("creating node {} with attributes {}", i, kw)
        self._T.add_node(i, **kw)
        return i

    @beartype
    def _active(self, pop: str) -> Node:
        """get the active (most recent) node for a population"""
        assert nx.is_forest(self._T), [(u.i, v.i) for u, v in self._T.edges()]
        for u in reversed(list(nx.topological_sort(self._T))):
            if pop in self.nodes[u]["block"]:
                return u

        raise ValueError(f"population {pop} not found in the event tree")

    @beartype
    def _merge_paths(self, p0: Path, p1: Path):
        "merge the blocks containing p0 and p1"
        bl0, bl1 = [next(s for s in self._paths if p in s) for p in (p0, p1)]
        if bl0 is bl1:
            return
        self._paths.remove(bl0)
        self._paths.remove(bl1)
        self._paths.add(bl0 | bl1)

    def _merge_nodes(self, x: Node, y: Node, t: Path, event: Event, rm=None) -> Node:
        """merge nodes x and y, optionally removing rm from the merged block set."""
        b = (
            self.nodes[x]["block"] | self.nodes[y]["block"]
        )  # new blocks, obtained by merging previous blocks
        if rm:
            b -= {rm}
        nn = self._add_node(block=b, t=t, event=event)
        for z in x, y:
            self._add_edge(z, nn)
        return nn

    @beartype
    def _pulse(self, source: str, dest: str, t: Path, prop_fun: Callable):
        """forward-in-time pulse from source into dest"""
        events = self.events
        u = self._active(dest)
        v = self._active(source)
        # there are two cases to consider depending on whether they are in the same
        # block or not
        if u is v:
            # same block, so we perform the pulse in one tensor contraction
            w = self._add_node(
                t=t,
                block=self.nodes[u]["block"],
                event=events.Pulse(source=source, dest=dest, prop_fun=prop_fun),
            )
            self._add_edge(u, w)
            return w
        else:
            # different blocks, so we model the pulse as an admixture followed by a
            # split2
            tr = unique_strs(self.nodes[u]["block"])[0]
            b = self.nodes[u]["block"] | {tr}
            w = self._add_node(
                block=b,
                t=t,
                event=events.Admix(
                    child=dest, parent1=tr, parent2=dest, prop_fun=prop_fun
                ),
            )
            self._add_edge(u, w)
            # now we need to merge the transient admixed population into the source
            # population
            ev = events.Split2(donor=tr, recipient=source)
            x = self._merge_nodes(w, v, t=t, event=ev, rm=tr)
            assert (
                self.nodes[x]["block"]
                == self.nodes[u]["block"] | self.nodes[v]["block"]
            )
            # identify which edge is which for traversal
            self.edges[w, x]["label"] = "donor"
            self.edges[v, x]["label"] = "recipient"
            # finally, rename the transient population to the destination population
            return x

    def _collapse_lifts(self):
        """Collapse successive lift events into a single event and merge identical times."""

        def simplify():
            # this function tries to find triples u -> v -> w which represent two successive
            # lifts, and then collapses them into a single lift event.
            for u, v in self._T.edges():
                succ = list(self._T.successors(v))
                if len(succ) == 0:
                    # root node
                    assert np.isinf(self._time(v))
                    continue
                else:
                    assert len(succ) == 1
                    w = succ[0]

                # eliminate pointless lifts
                ev = self.edges[u, v].get("event")
                if isinstance(ev, self.events.Lift) and self._time(u) == self._time(v):
                    assert self.nodes[u]["t"] == self.nodes[v]["t"]
                    assert self.nodes[u]["block"] == self.nodes[v]["block"]
                    assert "event" not in self.nodes[v]
                    del self.edges[u, v]["event"]
                    return False

                # if v does anything, we can't collapse it
                if self.nodes[v].get("event"):
                    continue

                # collapse the two lifts into a single lift
                ev = self.events.Lift()
                kw = {}
                if "label" in self._T.edges[v, w]:
                    kw["label"] = self._T.edges[v, w]["label"]
                assert not (
                    (self._T.edges[u, v].keys() & self._T.edges[v, w].keys())
                    - {"event"}
                )
                self._T.remove_node(v)
                self._add_edge(u, w, event=ev, **kw)
                logger.debug("removed node {} and collapsed lifts into {}", v, w)
                # repeat the process until there are no more successive lifts
                return False

            # we did not find any successive lifts, so we are done
            return True

        # repeat the process until there are no more successive lifts
        while not simplify():
            pass

    def draw(self, filename: str = None):
        """Draw the event tree."""
        import matplotlib.pyplot as plt
        from networkx.drawing.nx_pydot import graphviz_layout, write_dot

        # label each node and edge with its event
        edge_labels = {}
        for u, v in self._T.edges():
            ev = self.edges[u, v].get("event")
            if ev:
                edge_labels[u, v] = str(ev)

        node_labels = {}
        for u in self._T.nodes:
            ev = self.nodes[u].get("event")
            if ev:
                node_labels[u] = str(ev)

        root = next(n for n in self._T if self._T.out_degree(n) == 0)
        R = self._T.reverse()

        pos = graphviz_layout(R, prog="twopi", root=root)

        nx.draw(
            R, pos, labels=node_labels, with_labels=True, arrows=True, node_size=0.1
        )
        nx.draw_networkx_edge_labels(R, pos, edge_labels=edge_labels, font_color="red")
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        C = self._T.copy()
        for u in C.nodes():
            if "event" in C.nodes[u]:
                # remove the event from the node label
                C.nodes[u]["label"] = str(C.nodes[u]["event"])
            else:
                assert C.in_degree(u) == 0
                C.nodes[u]["label"] = next(iter(C.nodes[u]["block"]))
        write_dot(C, filename + ".dot" if filename else "event_tree.dot")
