Path = tuple[str | int, ...]


def get_path(params, path: Path) -> float:
    for i in path:
        params = params[i]
    return float(params)


def set_path(params, path: Path, value):
    for i in path[:-1]:
        params = params[i]
    params[path[-1]] = value


def bind(demo: dict, params: dict[Path, float]):
    ret = dict(demo)
    for path, val in params:
        set_path(ret, path, val)
    return ret
