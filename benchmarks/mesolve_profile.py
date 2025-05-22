import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from scipy.stats import unitary_group
import time
import json
from itertools import product


def benchmark_qutip_evolve(num_photons, num_modes=4, num_runs=1, num_points=1):
    """Benchmark a single mesolve() call in QuTiP for given photon count."""

    dim = 2 ** num_photons
    dims = [[2]*num_photons, [2]*num_photons]
    # Initial state: |...1⟩
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[-1] = 1.0
    rho0 = Qobj(np.outer(psi0, psi0.conj()), dims=dims)
    # psi0 = Qobj(psi0, dims=[dims[0], [1]])

    # eta_list = [1 - np.exp(-1j * 2 * np.pi * n / (num_photons + 1)) for n in range(num_photons + 1)]
    # v_configs = list(product(eta_list, repeat=num_modes))
    virtual_configs = product([1, 0], repeat=num_modes)

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(num_runs):
        # Unitary matrix for jump operator mixing
        unitary = unitary_group.rvs(num_modes)

        # Annihilation operators
        input_ops = [tensor([destroy(2) if i == j else qeye(2) for j in range(num_photons)]) for i in range(num_photons)]

        # Construct jump operators using unitary mixing
        jump_ops = []
        for i in range(num_modes):
            op = 0
            for j in range(num_photons):
                op += unitary[i, j] * input_ops[j]
            jump_ops.append(sprepost(op, op.dag()))

        # No Hamiltonian
        H = qeye(dims[0])

        # Time evolution
        tlist = [0, 8]
        for i, cfg in enumerate(virtual_configs):
            if i < num_points:
                jmps = [z * j for z, j in zip(cfg, jump_ops)]
                result = mesolve(H, rho0, tlist, c_ops=input_ops + jmps, e_ops=[])
                # result = sesolve(H, psi0, tlist)
    t1 = time.perf_counter()

    duration = t1 - t0
    return duration / num_runs, dim


def run_benchmarks(min_photons=1, max_photons=10, num_runs=1, num_points=1):
    results = []
    for n in reversed(range(min_photons, max_photons + 1)):
        try:
            print(f"Running QuTiP evolve for {n} qubits...")
            duration, hilbert_dim = benchmark_qutip_evolve(n, n + 1, num_runs=num_runs, num_points=num_points)
            print(f"  Time: {duration:.3f}s | Hilbert space size: {hilbert_dim}")
            results.append((n, hilbert_dim, duration))
        except Exception as e:
            print(f"  Failed for {n} photons: {e}")
            break
    return results


if __name__ == "__main__":
    results = run_benchmarks(min_photons=1, max_photons=10, num_runs=10, num_points=1)

    with open('data/results_qutip.json', 'w') as f:
        json.dump(results, f)

    photon_counts, hilbert_sizes, times = zip(*results)
    plt.figure()
    plt.plot(photon_counts, times, marker='o')
    plt.yscale('log')
    plt.title("QuTiP mesolve Time vs Photon Count")
    plt.xlabel("Number of Photons (Qubits)")
    plt.ylabel("Evolution Time (s, log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/qutip_evolve_benchmark.png")
    print("Plot saved to qutip_evolve_benchmark.png")
