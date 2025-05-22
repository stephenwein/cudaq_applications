from zpgenerator import *
import matplotlib.pyplot as plt
import time
import json


def benchmark_zpg(num_photons, num_modes=4, num_runs=1, num_points=1):
    t0 = time.perf_counter()
    for i in range(num_runs):
        p = Processor()
        p.add(list(range(num_photons)), Source.fock(1))
        p.add(0, Circuit.haar_random(num_modes))
        if num_points == 1:
            p.add(0, Detector.vacuum())
        else:
            p.add(0, Detector.pnr(num_points - 1))

        result = p.probs()
    t1 = time.perf_counter()

    duration = t1 - t0
    return duration / num_runs, 2 ** num_photons


def run_benchmarks(min_photons=1, max_photons=10, num_runs=1, num_points=1):
    results = []
    for n in reversed(range(min_photons, max_photons + 1)):
        try:
            print(f"Running probs() for {n} qubits...")
            duration, hilbert_dim = benchmark_zpg(n, n + 1, num_runs=num_runs, num_points=num_points)
            print(f"  Time: {duration:.3f}s | Hilbert space size: {hilbert_dim}")
            results.append((n, hilbert_dim, duration))
        except Exception as e:
            print(f"  Failed for {n} photons: {e}")
            break
    return results


if __name__ == "__main__":
    results = run_benchmarks(min_photons=1, max_photons=9, num_runs=1, num_points=16)

    with open('data/results_zpg.json', 'w') as f:
        json.dump(results, f)

    # Plot
    photon_counts, hilbert_sizes, times = zip(*results)
    plt.figure()
    plt.plot(photon_counts, times, marker='o')
    plt.yscale('log')
    plt.title("ZPGenerator probs() Time vs Photon Count")
    plt.xlabel("Number of Photons (Qubits)")
    plt.ylabel("Evolution Time (s, log scale)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("data/zpg_probs_benchmark.png")
    print("Plot saved to zpg_probs_benchmark.png")
