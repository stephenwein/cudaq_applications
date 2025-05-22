import cudaq
from cudaq.operator import *
import cupy as cp
import numpy as np
from scipy.stats import unitary_group
import time
import matplotlib.pyplot as plt
import json
from itertools import product


def benchmark_evolve(num_photons, num_modes=4, num_runs=1, num_points=1):
    """Benchmark a single evolve() call for given photon count."""
    cudaq.set_target("dynamics")

    dim = 2 ** num_photons
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[-1] = 1.0  # excited state

    rho0 = cudaq.State.from_data(np.outer(psi0, psi0.conj()))
    dimensions = {i: 2 for i in range(num_photons)}

    # eta_list = [1 - np.exp(-1j * 2 * np.pi * n / (num_photons + 1)) for n in range(num_photons + 1)]
    # v_configs = list(product(eta_list, repeat=num_modes))
    virtual_configs = product([1, 0], repeat=num_modes)

    # Benchmark evolve
    t0 = time.perf_counter()
    for _ in range(num_runs):
        # Operators
        unitary = np.array(unitary_group.rvs(num_modes), dtype=np.complex64)
        input_ops = [operators.annihilate(i) for i in range(num_photons)]
        jump_ops = [sum(unitary[i][j] * op for j, op in enumerate(input_ops)) for i in range(num_modes)]
        hamiltonian = sum(spin.z(i) for i in range(num_photons))
        tlist = cudaq.Schedule([0, 8], ["t"])

        for i, cfg in enumerate(virtual_configs):
            if i < num_points:
                result = cudaq.evolve(hamiltonian,
                                      dimensions,
                                      tlist,
                                      rho0,
                                      collapse_operators=input_ops,
                                      jump_operators=jump_ops,
                                      virtual_configuration=cfg)
    t1 = time.perf_counter()

    duration = t1 - t0
    return duration / num_runs, 2 ** num_photons


def run_benchmarks(min_photons=1, max_photons=10, num_runs=1, num_points=1):
    results = []
    for n in reversed(range(min_photons, max_photons + 1)):
        try:
            print(f"Running evolve() for {n} qubits...")
            duration, hilbert_dim = benchmark_evolve(n, n + 1, num_runs=num_runs, num_points=num_points)
            print(f"  Time: {duration:.3f}s | Hilbert space size: {hilbert_dim}")
            results.append((n, hilbert_dim, duration))
        except Exception as e:
            print(f"  Failed for {n} photons: {e}")
            break
    return results


if __name__ == "__main__":
    results = run_benchmarks(min_photons=1, max_photons=11, num_runs=10, num_points=1)

    with open('data/results_cudaq.json', 'w') as f:
        json.dump(results, f)

    # Plot
    photon_counts, hilbert_sizes, times = zip(*results)
    plt.figure()
    plt.plot(photon_counts, times, marker='o')
    plt.yscale('log')
    plt.title("CUDA-Q evolve() Time vs Photon Count")
    plt.xlabel("Number of Photons (Qubits)")
    plt.ylabel("Evolution Time (s, log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/cudaq_evolve_benchmark.png")
    print("Plot saved to cudaq_evolve_benchmark.png")
