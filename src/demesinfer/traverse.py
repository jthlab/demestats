"event tree traversals"

from collections.abc import Callable

from .event_tree import EventTree, Node


def traverse(
    et: EventTree,
    init_state: dict,
    # FIXME i am too lazy to figure out what are the corrct
    # type annotations
    node_callback: Callable,
    lift_callback: Callable,
    _fuse_lifts: bool,
) -> dict[Node, T]:
    """Traverse the event tree from the leaves upward and apply
    callbacks to nodes and edges.

    Args:
        et: The event tree to traverse.
        init_state: Initial state for each leaf node.
        node_callback: Callback function for nodes.
        lift_callback: Callback function for lifting between nodes.

    Returns:
        dict: A dictionary mapping nodes to their final states after traversal.
    """
    ret = {}
    states = dict(init_state)
    T = et.T

    def get_parent(node):
        parents = list(T.successors(node))
        assert len(parents) <= 1, "Node should have at most one parent."
        if parents == []:
            return None
        return parents[0]

    def get_children(node):
        children = list(T.predecessors(node))
        assert len(children) <= 2, "Node should have at most two children."
        return children

    states = dict(init_state)

    # fifo queue for processing nodes
    q = list(et.leaves)

    while q:
        # process children, transition upwards, and add the parent to the queue
        node = q.pop(0)
        children = get_children(node)
        match len(children):
            case 0:
                # leaf node, no event to process
                continue
            case 1:
                # single child, just update the state
                child = children[0]
                state = node_callback(node, child_state=states[child])
            case 2:
                # multiple children, need to aggregate states
                kw = {}
                for child in children:
                    # this label is guaranteed to exist for nodes that have multiple children
                    label = T.edges[child, node]["label"]
                    kw[label + "_state"] = states[child]
                state = node_callback(node, **kw)
            case _:
                raise ValueError("Node has more than two children, cannot process.")

        # now lift the node to just before its parent node
        # if _fuse_lifts, then we don't stop until we meet a node with
        # multiple children
        t0 = T.nodes[node]["t"]
        parent = get_parent(node)
        # this is the root node
        if parent is None:
            return states
        while len(get_children(parent)) == 1 and _fuse_lifts:
            parent = get_parent(parent)
        t1 = T.nodes[parent]["t"]
        states[node] = lift_callback(state, t0, t1)
        q.append(parent)
    return states
