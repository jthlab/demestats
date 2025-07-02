from demesinfer.event_tree import EventTree


def test_event_tree(demo):
    EventTree(demo)


def test_draw(demo):
    et = EventTree(demo)
    et.draw(filename="test_event_tree.pdf")
