import cudaq
from cudaq import State, Schedule, evolve, spin
from cupy import asnumpy
import numpy as np
from qutip import Qobj, ptrace
from itertools import product
from tqdm import tqdm
import cProfile, pstats, io
from circuits import CyclicInterferometer


def main(n_emitters, unitary, final_time=6.0, dephasing=0.0, detuning=0.0):
    # 1) Target
    cudaq.set_target("dynamics")

    # 2) Schedule
    t_initial, t_final = 0.0, final_time
    schedule = Schedule([t_initial, t_final], ["t"])

    # 3) System layout (3n qubits: n auxiliary, n×(photonic, spin))
    n_qubits = 3 * n_emitters
    n_modes = 2 * n_emitters
    auxiliaries = list(range(n_emitters))
    photons = list(range(n_emitters, 3 * n_emitters, 2))
    spins = list(range(n_emitters + 1, 3 * n_emitters, 2))
    dimensions = {i: 2 for i in range(n_qubits)}

    # 4) Hamiltonian: Zeeman + detuning
    g_factor, B_x = 2.0, np.pi / t_final / 8
    H_magnetic = g_factor * B_x * sum(spin.x(i) for i in spins) / 2
    H_detuning = sum(i * (detuning / (n_emitters - 1)) * spin.z(j) for i, j in enumerate(photons))
    H = H_detuning + H_magnetic

    # 5) Collapse ops: σ⁻ on each photonic qubit and σZ for pure dephasing
    decay_ops = [spin.minus(i) for i in photons]
    dephasing_ops = [np.sqrt(dephasing) * spin.z(i) for i in photons]
    collapse_ops = decay_ops + dephasing_ops

    # 6) Build spin projectors and 2n spin‐resolved jump operators
    P0 = [(spin.i(i) + spin.z(i)) * 0.5 for i in spins]
    P1 = [(spin.i(i) - spin.z(i)) * 0.5 for i in spins]

    jump_ops_local = [[P0[i] * spin.minus(j), P1[i] * spin.minus(j)] for i, j in enumerate(photons)]
    jump_ops_local = [j for jmps in jump_ops_local for j in jmps]

    # 7) Detector jump ops via a 2n×2n unitary mixing of the 2n σ⁻’s
    jump_ops = [sum(unitary[i, j] * jump_ops_local[j] for j in range(n_modes)) for i in range(n_modes)]

    # 8) build initial entangled state for Choi matrix simulation
    dim = 2 ** n_qubits
    psi = np.zeros(dim, dtype=np.complex128)

    # loop over aux bits and apply CNOT gates
    amp = (1 / np.sqrt(2)) ** n_emitters
    for b in product([0, 1], repeat=n_emitters):
        bits = list(b) + [bit for i in range(n_emitters) for bit in (0, b[i])]
        idx = int("".join(map(str, bits)), 2)
        psi[idx] = amp

    rho0 = State.from_data(np.outer(psi, psi.conj()))

    # 9) Sweep virtual configs
    virtual_configs = product([0, 1], repeat=n_modes)
    rho_tensor = np.empty((2,) * n_modes, dtype=object)

    for cfg in tqdm(virtual_configs):
        res = evolve(
            H, dimensions, schedule, rho0,
            collapse_operators=collapse_ops,
            jump_operators=jump_ops,
            virtual_configuration=[1 - z for z in cfg],
            store_intermediate_results=False)

        rho_q = Qobj(asnumpy(res.final_state()), dims=[[2] * n_qubits, [2] * n_qubits])
        reduced = ptrace(rho_q, auxiliaries + spins)  # partial trace out photonic qubits
        dm = [2] * n_emitters
        reduced = Qobj(reduced, dims=[[dm, dm], [dm, dm]], superrep='Choi')  # reshape into superoperator (Choi)
        rho_tensor[cfg] = reduced

    # 10) Contract with M = [[1, 0], [-1, 1]] along each axis
    M = np.array([[1, 0], [-1, 1]])
    choi_matrices = rho_tensor
    for ax in range(n_modes):
        choi_matrices = np.moveaxis(np.tensordot(M, choi_matrices, axes=(1, ax)), 0, ax)

    return choi_matrices


if __name__ == '__main__':

    n_emitters = 3
    unitary = CyclicInterferometer(n_emitters).compute_unitary()

    pr = cProfile.Profile()
    pr.enable()
    choi_matrices = main(n_emitters, unitary, final_time=6.0, dephasing=0.05, detuning=0.05)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(10)
    print(s.getvalue())

    it = np.nditer(choi_matrices, flags=['multi_index', 'refs_ok'])
    for rho in it:
        state = rho.item()
        prob = state.tr().real
        print(f"{it.multi_index} → {prob:.6f}")

    np.save('choi_matrices', choi_matrices)
