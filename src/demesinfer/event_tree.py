import math
import pathlib
from collections import UserList, defaultdict
from collections.abc import Collection, Sequence, Set
from copy import deepcopy
from enum import Enum
from functools import cached_property, total_ordering
from itertools import count
from types import ModuleType

import demes
import jax.numpy as jnp
import networkx as nx
import numpy as np
from beartype.typing import Callable, Iterable
from jaxtyping import Float, ScalarLike
from loguru import logger
from scipy.cluster.hierarchy import DisjointSet

import demesinfer.events
from demesinfer.rescale import rescale_demo

from .path import Path, get_path, set_path
from .util import unique_strs

Variable = Set[Path]


class _FSList(UserList):
    def add(self, *items: Path):
        self.data.append(frozenset(items))


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


def _collapse_noops(T: nx.DiGraph):
    events = demesinfer.events
    COLLAPSE_TYPES = (
        events.MigrationStart,
        events.MigrationEnd,
        events.Epoch,
        events.NoOp,
    )

    def f():
        # find a NoOp node and remove it
        for n in T.nodes:
            if (
                T.out_degree(n) == 1
                and T.in_degree(n) == 1
                and isinstance(T.nodes[n]["event"], COLLAPSE_TYPES)
            ):
                # remove the node
                logger.trace("removing noop node {}", n)
                (u,) = T.predecessors(n)
                (v,) = T.successors(n)
                kw = {}
                if "label" in T.edges[n, v]:
                    kw["label"] = T.edges[n, v]["label"]
                T.remove_node(n)
                T.add_edge(u, v, **kw)
                # return True to indicate that we found a NoOp node
                return True
        return False

    while f():
        pass


def _all_events(demo: demes.Graph) -> Iterable[dict]:
    """Iterate over all events in the demes graph"""
    d = demo.asdict()
    for i, deme in enumerate(d["demes"]):
        name = deme["name"]
        for j, e in enumerate(deme["epochs"]):
            # size change events
            time_path = ("demes", i, "epochs", j, "end_time")
            yield dict(
                t=time_path,
                pop=name,
                size_function=e["size_function"],
                ev=EventType.EPOCH,
                i=j,
            )
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
    """
    Build an event tree from a demes graph.

    Parameters
    ----------
        demo : demes.Graph
            a ``demes`` graph
        events : ModuleType, optional
            a module containing event classes
        _merge_contemp : bool, optional
            Boolean necessary for tree creation

    Returns:
        EventTree: event tree built from a demes graph

    Notes
    -----
    From a user perspective, understanding the underlying structure of an EventTree
    is not necessary. The only functions that a user would use is ``EventTree.variables``
    which lists out all the parameters/variables in the event tree and ``EventTree.variable_for`` which
    allows a user to input a ``demes`` path and find the associated EventTree variable. 
    
    Example:
    ::
        # EventTree.variables
        et = EventTree(demo.to_demes())
        print(et.variables)

        # EventTree.variable_for, example taken from tutorial
        parameters = [
            ('demes', 0, 'epochs', 0, 'end_size'), # The ancestral population size
            ('migrations', 0, 'rate'), # Rate of migration from P0 to P1
            ('demes', 0, 'epochs', 0, 'end_time') # Time of divergence
        ]

        momi3_parameters = [et.variable_for(param) for param in parameters]

    Please refer to the tutorial for a specific example, the above provided codes are just outlines of how to call on the functions.
    
    See Also
    --------
    demesinfer.event_tree.EventTree.variable_for
    demesinfer.event_tree.EventTree.variables
    """

    def __init__(
        self,
        demo: demes.Graph,
        events: ModuleType = demesinfer.events,
        _merge_contemp: bool = False,
    ):
        # store passed parameters
        self._demo = demo
        self._events = events
        self._merge_contemp = _merge_contemp

        # data structures for tree
        self._T = nx.DiGraph()
        self._i = count(0)

        self._demodict = demo.asdict()

        # tree creation
        self._init_leaves()
        self._build_tree()
        self._decorate_times()
        self._check()

    @property
    def T(self):
        """Get the event tree."""
        return self._T

    @cached_property
    def T_reduced(self):
        """Get the event tree for traversal."""
        ret = deepcopy(self._T)  # keep a copy of the full tree
        # _collapse_noops(ret)
        return ret

    @property
    def demo(self):
        """Get the demes graph."""
        return self._demo

    @property
    def demodict(self):
        return self._demodict

    def _init_demodict(self):
        """Get the demes graph as a dictionary."""

    @property
    def scaling_factor(self) -> ScalarLike:
        """
        The scaling factor used for the demo.
        This is used to rescale the demo parameters.
        """
        return self.demo.demes[0].epochs[0].start_size

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
    def leaves(self) -> dict[str, Node]:
        ret = {}
        for n in self.leaves_below(self.root):
            (pop,) = self.nodes[n]["block"]
            ret[pop] = n
        return ret

    def variable_for(self, path: Path) -> Variable:
        """
        Get the EventTree variable corresponding to ``demes`` path

        Parameters
        ----------
            path : Path
                a ``demes`` path

        Returns:
            Variable: The associated EventTree variable given a path

        Notes
        -----
        To use this function you must create an EventTree object. 
        For more details see the EventTree API. 
        
        See Also
        --------
        demesinfer.event_tree.EventTree
        """
        for v in self.variables:
            if isinstance(v, tuple):
                if v == path:
                    return v
            elif path in v:
                return v
        raise ValueError(f"path {path} is not a variable")

    def _decorate_times(self):
        """Decorate each node with its time value."""
        # to determine the time identities, traverse the full tree, and identify any times which result in a zero lift
        times = DisjointSet()
        for u, v in self.edges:
            p_u, p_v = [self.nodes[n]["t"] for n in (u, v)]
            t_u, t_v = [self.get_time(n) for n in (u, v)]
            times.add(p_u)
            times.add(p_v)
            if t_u == t_v:
                times.merge(p_u, p_v)

        self._times = times = sorted(times.subsets())
        for n in self._T.nodes:
            t_path = self.nodes[n]["t"]
            i = next(i for i, s in enumerate(times) if t_path in s)
            self._T.nodes[n]["ti"] = i

        self._demodict["_times"] = jnp.array(
            [self.get_path(next(iter(s))) for s in times]
        )

    def constant_growth_in(self, pops: Iterable[str], t0i: int, t1i: int) -> bool:
        demo = self.demodict
        t0, t1 = [next(iter(self.times[ti])) for ti in (t0i, t1i)]
        a = get_path(demo, t0)
        b = get_path(demo, t1)
        for pop in pops:
            d = next(d for d in demo["demes"] if d["name"] == pop)
            start_time = d["start_time"]
            for e in d["epochs"]:
                v = start_time
                u = e["end_time"]
                # check if [a, b] and [u, v] overlap
                if max(a, u) < min(b, v):
                    if e["size_function"] != "constant":
                        return False
                start_time = u
        return True

    @property
    def times(self) -> list[Set[Path]]:
        """Get the time variables in the event tree."""
        return self._times

    @cached_property
    def variables(self) -> Sequence[Variable]:
        """
        List out **all** of the EventTree variables corresponding to a ``demes`` graph

        Returns:
            Sequence[Variable]: **All** associated EventTree variables given a ``demes`` graph

        Notes
        -----
        To use this function you must create an EventTree object. 
        For more details see the EventTree API. 
        
        See Also
        --------
        demesinfer.event_tree.EventTree
        """
        ret = defaultdict(_FSList)
        dd = self.demodict

        for i, d in enumerate(dd["demes"]):
            assert "end_time" not in d
            for j, e in enumerate(d["epochs"]):
                path0 = ("demes", i, "epochs", j, "start_size")
                path1 = ("demes", i, "epochs", j, "end_size")
                if e["size_function"] == "constant":
                    ret["sizes"].add(path0, path1)
                else:
                    for p in path0, path1:
                        ret["sizes"].add(p)

            for j, _ in enumerate(d.get("proportions", [])):
                ret["proportions"].add(("demes", i, "proportions", j))

        for i, m in enumerate(dd["migrations"]):
            ret["rates"].add(("migrations", i, "rate"))

        # proportions
        for i, p in enumerate(dd["pulses"]):
            rhos = []
            for j, _ in enumerate(p["proportions"]):
                ret["proportions"].add(("pulses", i, "proportions", j))

        # convert single-element sets to single elements
        assert np.isinf(self.get_path(next(iter(self.times[-1]))))
        # do not include the final time (infinity)
        ret["times"] = [frozenset(s) for s in self.times][:-1]
        return sum(map(list, ret.values()), [])

    def bind(
        self, params: dict[Set[Path] | Path, ScalarLike], rescale: bool = False
    ) -> dict:
        # bind the parameters to the event tree
        if not params.keys() <= set(self.variables):
            raise ValueError(
                "The parameters must be a subset of the event tree variables."
                " Got: "
                f"{params.keys() - set(self.variables)}"
            )
        ret = deepcopy(self.demodict)
        for k, v in params.items():
            if isinstance(k, tuple):
                k = frozenset([k])
            for p in k:
                set_path(ret, p, v)
        if rescale:
            # rescale the demo parameters
            ret = rescale_demo(ret, self.scaling_factor)
        times = []
        for paths in self.times:
            path = next(iter(paths))
            val = get_path(ret, path)
            times.append(val)
        ret["_times"] = jnp.array(times)
        return ret

    def _init_leaves(self):
        # Initialize leaf nodes for each population
        leaves = defaultdict(list)
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
                    len(deme.epochs) - 1,
                    "end_time",
                ),  # time of the population start
                block=frozenset([deme.name]),
                event=self._events.PopulationStart(),
            )
            t = self._demo.demes[j].end_time
            leaves[t].append(self._add_node(**kw))

        if not self._merge_contemp:
            return

        # if merging contemporaneous populations, we need to merge events for each
        # population at each time point
        for t, nodes in leaves.items():
            x = nodes.pop()
            while nodes:
                y = nodes.pop()
                # merge the two nodes into one
                z = self._merge_nodes(
                    x, y, t=self.nodes[y]["t"], event=self._events.Merge()
                )
                self.edges[x, z]["label"] = "pop1"
                self.edges[y, z]["label"] = "pop2"
                x = z

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
                    w = self._merge_nodes(u, v, t=t, event=events.Merge())
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

            elif d["ev"] == EventType.EPOCH:
                # epoch events are just lifting events, so we lift the node u to the
                # time of the epoch.
                v = self._add_node(
                    t=t,
                    block=self.nodes[u]["block"],
                    event=events.Epoch(is_constant=d["size_function"] == "constant"),
                )
                self._add_edge(u, v)
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
        self._check()

    @property
    def nodes(self):
        """Get the node with id i."""
        return self._T.nodes

    @property
    def edges(self):
        return self._T.edges

    def get_path(self, p: Path) -> ScalarLike:
        """Get the value of path p in the event tree."""
        return get_path(self.demodict, p)

    def get_var(self, v: Variable) -> ScalarLike:
        """Get the value of variable v in the event tree."""
        assert v in self.variables
        if isinstance(v, Set):
            v = next(iter(v))  # get the first element of the set
        return self.get_path(v)

    def get_time(self, node: Node) -> ScalarLike:
        return self.get_path(self.nodes[node]["t"])

    def _check(self):
        assert nx.is_tree(self._T)  # sanity check.
        # check that the block each node comprises the populations beneath that node
        root = next(n for n in self._T if self._T.out_degree(n) == 0)
        assert len(self.nodes[root]["block"]) == 1

    def _add_edge(self, u: Node, v: Node):
        succ = list(self._T.successors(u))
        assert not succ, (u, v, succ)
        logger.trace("adding edge {} -> {}", u, v)
        self._T.add_edge(u, v)

    def _remove_edge(self, u: Node, v: Node):
        """Remove the edge u -> v from the tree."""
        assert self._T.has_edge(u, v), (u, v)
        logger.trace("removing edge {} -> {}", u, v)
        self._T.remove_edge(u, v)

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
        logger.trace("creating node {} with attributes {}", i, kw)
        self._T.add_node(i, **kw)
        return i

    def _active(self, pop: str) -> Node:
        """get the active (most recent) node for a population"""
        # FIXME this is super inefficient but it hardly matters
        for u in reversed(list(nx.topological_sort(self._T))):
            if pop in self.nodes[u]["block"]:
                return u

        raise ValueError(f"population {pop} not found in the event tree")

    def _merge_paths(self, p0: Path, p1: Path):
        "merge the blocks containing p0 and p1"
        bl0, bl1 = [next(s for s in self._paths if p in s) for p in (p0, p1)]
        if bl0 is bl1:
            return
        self._paths.remove(bl0)
        self._paths.remove(bl1)
        self._paths.add(bl0 | bl1)

    def _merge_nodes(
        self, x: Node, y: Node, t: Path, event: demesinfer.events.Event, rm=None
    ) -> Node:
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

    def draw(self, filename: str | pathlib.Path = None):
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
        # C = self._T.copy()
        # for u in C.nodes():
        #     if "event" in C.nodes[u]:
        #         # remove the event from the node label
        #         C.nodes[u]["label"] = str(C.nodes[u]["event"])
        #     else:
        #         assert C.in_degree(u) == 0
        #         C.nodes[u]["label"] = next(iter(C.nodes[u]["block"]))
        # write_dot(C, filename + ".dot" if filename else "event_tree.dot")
