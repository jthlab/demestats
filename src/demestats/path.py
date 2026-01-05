from copy import deepcopy

from jaxtyping import ScalarLike

Path = tuple[str | int, ...]


def get_path(params, path: Path) -> ScalarLike:
    for i in path:
        params = params[i]
    return params


def set_path(params, path: Path, value: ScalarLike):
    for i in path[:-1]:
        params = params[i]
    params[path[-1]] = value


def bind(demo: dict, params: dict[Path, ScalarLike]) -> dict:
    ret = deepcopy(demo)
    for path, val in params.items():
        set_path(ret, path, val)
    return ret
