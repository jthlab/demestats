from demesinfer.event_tree import EventTree


def test_event_tree(demo):
    EventTree(demo)


def test_draw(demo, tmp_path):
    et = EventTree(demo)
    path = tmp_path / "test_event_tree.pdf"
    et.draw(filename=path)
    assert path.exists()
