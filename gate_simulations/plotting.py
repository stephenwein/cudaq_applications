import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import qutip as qt


def generate_pauli_labels(n_qubits):
    """Return list of n-qubit Pauli string labels."""
    paulis = ['I', 'X', 'Y', 'Z']
    return [''.join(p) for p in product(paulis, repeat=n_qubits)]


def plot_chi_heatmap(chi, title, filename):
    n = int(np.log2(np.sqrt(chi.shape[0])))
    labels = generate_pauli_labels(n)

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(chi, cmap='seismic', vmin=-np.max(np.abs(chi)), vmax=np.max(np.abs(chi)))

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_chi_heatmap_im(chi_matrix, title="Chi Matrix (Imaginary Part)", filename="chi_heatmap_im.png"):
    if hasattr(chi_matrix, 'full'):
        chi = np.imag(chi_matrix.full())
    else:
        chi = np.imag(chi_matrix)
    plot_chi_heatmap(chi, title, filename)


def plot_chi_heatmap_re(chi_matrix, title="Chi Matrix (Real Part)", filename="chi_heatmap_re.png"):
    if hasattr(chi_matrix, 'full'):
        chi = np.real(chi_matrix.full())
    else:
        chi = np.real(chi_matrix)
    plot_chi_heatmap(chi, title, filename)


choi_matrices = np.load('data/cz_noisy/choi_matrices.npy', allow_pickle=True)


# channel = choi_matrices[1, 0, 1, 0]
#
# norm = channel.tr()
# print(norm)
# channel = qt.choi_to_super(channel / norm)

# rx = qt.to_super(qt.qip.circuit.rx(7 * np.pi / 4, 3, 0) *
#                  qt.qip.circuit.rx(7 * np.pi / 4, 3, 1) *
#                  qt.qip.circuit.rx(7 * np.pi / 4, 3, 2))
# sa = qt.to_super(qt.qip.circuit.s_gate(3, 0) *
#                  qt.qip.circuit.s_gate(3, 1) *
#                  qt.qip.circuit.s_gate(3, 2))
#
# channel = qt.to_chi(sa * rx * channel)
# channel_target = qt.Qobj(np.sign(np.real(channel.tidyup(atol=0.5))), dims=channel.dims)
# channel_target = channel_target / qt.chi_to_choi(channel_target).tr()
#
# d = 2**3
# fidelity = ((channel_target.dag() * channel).tr() + d) / (d * (d + 1))
# print(fidelity)

# plot_chi_heatmap_re(channel)
# plot_chi_heatmap_im(channel)

#  Extra analysis for n=2

# 11) Feedforward corrections
sa = qt.qip.circuit.s_gate(2, 0)
sb_dag = qt.qip.circuit.s_gate(2, 1).dag()
correction_1 = qt.to_super(sa * sb_dag)
correction_2 = correction_1.dag()

successes = {(0, 1, 0, 1): correction_1,
             (0, 1, 1, 0): correction_2,
             (1, 0, 0, 1): correction_2,
             (1, 0, 1, 0): correction_1}

zz = qt.to_super(qt.qip.circuit.z_gate(2, 0) * qt.qip.circuit.z_gate(2, 1))
ii = qt.to_super(qt.qeye([[2, 2], [2, 2]]))  # identity superoperator
retries = {(1, 0, 0, 0): ii, (0, 1, 0, 0): ii, (0, 0, 1, 0): zz, (0, 0, 0, 1): zz}

# 12) Apply corrections, normalize, and compose total Chi matrix
rx = qt.to_super(qt.qip.circuit.rx(7 * np.pi / 4, 2, 0) * qt.qip.circuit.rx(7 * np.pi / 4, 2, 1))
# channel_super = sum(choi_matrices[k].tr() * U * rx * qt.choi_to_super(choi_matrices[k] / choi_matrices[k].tr())
#               for k, U in successes.items())
channel_super = sum(choi_matrices[k].tr() * U * rx * qt.choi_to_super(choi_matrices[k] / choi_matrices[k].tr())
              for k, U in retries.items())

norm = qt.super_to_choi(channel_super).tr()
channel_super = channel_super / norm

print('Gate success probability: ', norm)

qt.qpt_plot_combined(qt.to_chi(channel_super), lbls_list=[['i', 'x', 'y', 'z']] * 2)
plt.savefig("chi_matrix.png")
#
# 13) Compute error chi matrix and average gate fidelity
cz = qt.qip.circuit.cz_gate(2, 0, 1)
ii = qt.qip.circuit.identity([2, 2])
print(qt.average_gate_fidelity(4*channel_super, ii))
#
# channel_err = qt.to_chi(qt.to_super(cz.dag()) * channel) / 16
# qt.qpt_plot_combined(channel_err, lbls_list=[['i', 'x', 'y', 'z']] * 2)
# plt.savefig("chi_error_matrix.png")
#
# plot_chi_heatmap_re(channel_chi)
# plot_chi_heatmap_im(channel_chi)
