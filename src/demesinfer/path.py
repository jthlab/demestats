Path = tuple[str | int, ...]


def get_path(params, path: Path):
    for i in path:
        params = params[i]
    return params
