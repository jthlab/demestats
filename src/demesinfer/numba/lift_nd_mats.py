import numpy as np
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
