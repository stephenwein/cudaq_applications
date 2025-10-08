from itertools import product
from numpy import zeros, complex128, sqrt, outer
from cudaq import State
from .utils import idx_from_digits_lsb


def choi_initial_state(n_emitters: int):
    # Choi input: optics in |1...1>, (aux_i,spn_i) Bell-tied
    n_qubits = 3 * n_emitters
    auxilia = list(range(n_emitters))
    photons = list(range(n_emitters, 3 * n_emitters, 2))
    spins = list(range(n_emitters + 1, 3 * n_emitters, 2))

    dim = 2 ** n_qubits
    psi = zeros(dim, dtype=complex128)
    amp = (1 / sqrt(2)) ** n_emitters
    for b in product([0, 1], repeat=n_emitters):
        digits = [0] * n_qubits
        for i, bi in enumerate(b):
            digits[auxilia[i]] = bi
            digits[photons[i]] = 1
            digits[spins[i]] = bi
        psi[idx_from_digits_lsb(digits)] = amp
    rho0 = State.from_data(outer(psi, psi.conj()))

    return rho0
