import matplotlib.pyplot as plt
import json

doc = 'single point'

with open('data/' + doc + '/results_cudaq.json') as f:
    results_cudaq = json.load(f)

with open('data/' + doc + '/results_qutip.json') as f:
    results_qutip = json.load(f)

with open('data/' + doc + '/results_qutip_ham.json') as f:
    results_qutip2 = json.load(f)

with open('data/' + doc + '/results_zpg.json') as f:
    results_zpg = json.load(f)

with open('data/' + doc + '/results_cudaq_lind.json') as f:
    results_cudaq2 = json.load(f)

with open('data/' + doc + '/results_cudaq_ham.json') as f:
    results_cudaq3 = json.load(f)

results_list = [results_cudaq, results_zpg, results_qutip, results_cudaq2, results_cudaq3, results_qutip2]
color_list = ['green', 'red', 'blue', 'orange', 'purple', 'cyan']
label_list = ['cudaq', 'zpg', 'qutip', 'cudaq-unconditional', 'cudaq-unitary', 'qutip-unitary']

# Plot
plt.figure()
for color, results, label in zip(color_list, results_list, label_list):
    photon_counts, hilbert_sizes, times = zip(*results)
    plt.plot(photon_counts, [t for t in times], marker='o', color=color, label=label)
plt.yscale('log')
plt.title("Single-point evaluation")
plt.xlabel("Number of Photons (Qubits)")
plt.ylabel("Evolution Time (s, log scale)")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.savefig("benchmark.png")

