import numpy as np
import scipy.sparse as sp
from jax.experimental.sparse import BCOO
from jaxtyping import jaxtyped
from numba import njit


@njit
def _flatten_idx(idx, base):
    d = idx.shape[0]
    x = 0
    for k in range(d):
        x = x * base + idx[k]
    return x


@njit
def _mats_worker(d, n, B, U_indices, U_data):
    base = n + 1
    S = base**d
    ptr = 0
    for flat_i in range(S):
        i = B[flat_i]
        for v in range(d):
            if i[v] > 0:
                for u in range(d):
                    if u != v and i[u] < n:
                        j = i.copy()
                        j[u] += 1
                        j[v] -= 1
                        flat_j = _flatten_idx(j, base)
                        U_indices[ptr, 0] = flat_i
                        U_indices[ptr, 1] = flat_j
                        U_indices[ptr, 2] = u
                        U_indices[ptr, 3] = v
                        U_data[ptr] = 1
                        ptr += 1
    return ptr


def mats(d, n):
    B = np.array(list(np.ndindex((n + 1,) * d))).reshape((n + 1,) * d + (d,))

    S = (n + 1) ** d
    nnz_max = S * d * (d - 1)
    U_indices = np.zeros((nnz_max, 4), dtype=np.int64)
    U_data = np.zeros(nnz_max, dtype=np.uint8)

    nnz = _mats_worker(d, n, B.reshape(-1, d), U_indices, U_data)
    U_data = U_data[:nnz]
    U_indices = U_indices[:nnz]
    U = BCOO((U_data, U_indices), shape=(S, S, d, d))
    U = U.reshape((n + 1,) * (2 * d) + (d, d))
    U = U.sort_indices()
    return dict(B=B, U=U)


import numpy as np
from jax.experimental.sparse import BCOO
from numba import njit


@njit
def _flatten_idx(idx, base):
    d2 = idx.shape[0]
    x = 0
    for k in range(d2):
        x = x * base + idx[k]
    return x


def mats_ccr(pop_names: tuple[str, ...], n: int):
    """
    Build sparse transition matrices for the CCR process.
    Returns per-migration-pair and per-deme coalescent matrices stored as csr matrices.
    """

    def empty_entry():
        return dict(rows=[], cols=[], data=[])

    d = len(pop_names)
    colors = ("red", "blue")
    axes = tuple((pop, color) for pop in pop_names for color in colors)
    ndim = len(axes)
    base = n + 1
    total_states = base**ndim
    shape = (base,) * ndim

    def flatten(idx):
        ret = 0
        for val in idx:
            ret = ret * base + val
        return ret

    migration_entries = {
        color: {
            (dest, src): empty_entry()
            for src in range(d)
            for dest in range(d)
            if dest != src
        }
        for color in colors
    }
    coalesce_entries = {color: [empty_entry() for _ in range(d)] for color in colors}
    cross_entries = [empty_entry() for _ in range(d)]

    for idx in np.ndindex(*shape):
        coord = list(idx)
        flat = flatten(coord)
        for deme in range(d):
            red_axis = 2 * deme
            blue_axis = red_axis + 1
            red_count = coord[red_axis]
            blue_count = coord[blue_axis]

            if red_count > 0:
                for dest in range(d):
                    if dest == deme:
                        continue
                    dest_axis = 2 * dest
                    if coord[dest_axis] >= n:
                        continue
                    new_coord = coord.copy()
                    new_coord[red_axis] -= 1
                    new_coord[dest_axis] += 1
                    tgt = flatten(new_coord)
                    entry = migration_entries["red"][(dest, deme)]
                    entry["rows"].extend((tgt, flat))
                    entry["cols"].extend((flat, flat))
                    entry["data"].extend((red_count, -red_count))

            if red_count >= 2:
                new_coord = coord.copy()
                new_coord[red_axis] -= 1
                tgt = flatten(new_coord)
                weight = red_count * (red_count - 1) / 2
                entry = coalesce_entries["red"][deme]
                entry["rows"].extend((tgt, flat))
                entry["cols"].extend((flat, flat))
                entry["data"].extend((weight, -weight))

            if blue_count > 0:
                for dest in range(d):
                    if dest == deme:
                        continue
                    dest_axis = 2 * dest + 1
                    if coord[dest_axis] >= n:
                        continue
                    new_coord = coord.copy()
                    new_coord[blue_axis] -= 1
                    new_coord[dest_axis] += 1
                    tgt = flatten(new_coord)
                    entry = migration_entries["blue"][(dest, deme)]
                    entry["rows"].extend((tgt, flat))
                    entry["cols"].extend((flat, flat))
                    entry["data"].extend((blue_count, -blue_count))

            if blue_count >= 2:
                new_coord = coord.copy()
                new_coord[blue_axis] -= 1
                tgt = flatten(new_coord)
                weight = blue_count * (blue_count - 1) / 2
                entry = coalesce_entries["blue"][deme]
                entry["rows"].extend((tgt, flat))
                entry["cols"].extend((flat, flat))
                entry["data"].extend((weight, -weight))

            cross_weight = red_count * blue_count
            if cross_weight:
                entry = cross_entries[deme]
                entry["rows"].append(flat)
                entry["cols"].append(flat)
                entry["data"].append(-cross_weight)

    def to_csr(entry):
        if not entry["data"]:
            return sp.csr_matrix((total_states, total_states))
        data = np.array(entry["data"], dtype=np.float64)
        rows = np.array(entry["rows"], dtype=np.int64)
        cols = np.array(entry["cols"], dtype=np.int64)
        return sp.csr_matrix((data, (rows, cols)), shape=(total_states, total_states))

    migration = {
        color: {(dest, src): to_csr(entry) for (dest, src), entry in entries.items()}
        for color, entries in migration_entries.items()
    }
    coalesce = {
        color: [to_csr(entry) for entry in entries]
        for color, entries in coalesce_entries.items()
    }
    cross = [to_csr(entry) for entry in cross_entries]

    return dict(
        axes=axes,
        pops=tuple(pop_names),
        size=total_states,
        shape=shape,
        migration=migration,
        coalesce=coalesce,
        cross=cross,
    )
