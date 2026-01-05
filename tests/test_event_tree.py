import pytest

from demestats.event_tree import EventTree
from demestats.path import get_path

from .demos import SingleDeme


def test_event_tree(demo):
    EventTree(demo)


@pytest.mark.skip
def test_draw(demo, tmp_path):
    et = EventTree(demo)
    path = tmp_path / "test_event_tree.pdf"
    et.draw(filename=path)
    assert path.exists()


def test_bind():
    demo, _ = SingleDeme.Constant().base()
    et = EventTree(demo)
    paths = [
        ("demes", 0, "epochs", 0, "start_size"),
        ("demes", 0, "epochs", 0, "end_size"),
    ]
    params = {paths[0]: 0.1}
    with pytest.raises(ValueError):
        demo = et.bind(params)
    params = {frozenset(paths): 0.1}
    demo = et.bind(params)
    assert get_path(demo, paths[0]) == get_path(demo, paths[1]) == 0.1
