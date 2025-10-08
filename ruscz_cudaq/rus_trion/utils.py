import numpy as np

def idx_from_digits_lsb(digs):
    """Flatten LSB-first bits to integer."""
    idx, mult = 0, 1
    for d in digs:
        idx += (1 if d else 0) * mult
        mult <<= 1
    return idx

def lsb_to_msb(rho, n_subs: int, dsub: int = 2):
    """Reorder density matrix from LSB-first to MSB-first subsystem order."""
    T = np.asarray(rho).reshape(*([dsub]*n_subs + [dsub]*n_subs))
    perm = list(range(n_subs-1, -1, -1)) + list(range(2*n_subs-1, n_subs-1, -1))
    return np.transpose(T, perm).reshape(dsub**n_subs, dsub**n_subs)

def block_auxspin_to_choi(R: np.ndarray, n_emitters: int) -> np.ndarray:
    """
    R after ptrace(aux+spin) in order [aux0..aux{n-1}, spn0..spn{n-1}] (ket|bra).
    Return as Choi with axes (spins | aux) on both ket & bra.
    """
    m = 2 * n_emitters
    T = np.asarray(R).reshape(*([2]*m + [2]*m))
    aux_k = list(range(0, n_emitters))
    spn_k = list(range(n_emitters, m))
    aux_b = [i + m for i in aux_k]
    spn_b = [i + m for i in spn_k]
    perm = spn_k + aux_k + spn_b + aux_b
    return np.transpose(T, perm).reshape(2**m, 2**m)

def ptrace(rho: np.ndarray, dims, keep):
    """
    Partial trace of an operator rho over all subsystems not in `keep`.
    - rho: (D,D) ndarray
    - dims: list/tuple of subsystem dims [d0, d1, ..., d_{N-1}]  (MSB-first)
    - keep: iterable of subsystem indices to keep (indices refer to dims)
    Returns: (D_keep, D_keep) ndarray
    """
    dims = list(dims)
    N = len(dims)
    keep = sorted(keep)
    drop = sorted([i for i in range(N) if i not in keep], reverse=True)
    T = rho.reshape(*dims, *dims)

    # Trace out dropped subsystems, descending
    for i in drop:
        bk = T.ndim // 2
        T = np.trace(T, axis1=i, axis2=bk + i)

    d_keep = int(np.prod([dims[i] for i in keep])) if keep else 1
    return T.reshape(d_keep, d_keep)