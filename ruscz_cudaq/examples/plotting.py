import numpy as np
import matplotlib.pyplot as plt
import qutip as qt

choi_matrices = np.load('choi_matrices.npy')
targets = np.load('choi_targets.npy')

sigma_list = np.linspace(0, 0.3, 30)
print(sigma_list[10])

pmap = qt.Qobj(choi_matrices[10][1,0,1,0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
pmap = pmap / pmap.tr()

targ = qt.Qobj(targets[1, 0, 1, 0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
targ = targ / targ.tr()

chi_err = qt.to_chi(qt.to_super(targ).inv() * qt.to_super(pmap))
print(chi_err)


pauli_errors = []
for choi in choi_matrices:
    pmap = qt.Qobj(choi[1, 0, 1, 0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
    pmap = pmap / pmap.tr()

    targ = qt.Qobj(targets[1, 0, 1, 0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
    targ = targ / targ.tr()

    chi_err = qt.to_chi(qt.to_super(targ).inv() * qt.to_super(pmap))
    pauli_errors.append(np.diag(chi_err / chi_err.tr()))

a  = np.char.array(list('ixyz'))
a = a[:, None]+a
a = [x for xs in a for x in xs]

sigmas = np.linspace(0, 0.3, 30)
pauli_errors = np.array(pauli_errors).transpose()
for lb, dat in zip(a[1:], pauli_errors[1:]):
    plt.plot(sigmas, dat, label=lb)
plt.legend()
plt.xlabel("Overhauser field strength, $\sigma_{OH}/\gamma$")
plt.ylabel("Stohastic pauli error")

plt.savefig("pauli.png")

pmap = qt.Qobj(choi_matrices[-1][1, 0, 1, 0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
pmap = pmap / pmap.tr()

targ = qt.Qobj(targets[1, 0, 1, 0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
targ = targ / targ.tr()

chi_err = qt.to_chi(qt.to_super(targ).inv() * qt.to_super(pmap))
chi_err = chi_err.data
chi_err[0,0] = 0
chi_err = qt.Qobj(chi_err, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='chi')

qt.qpt_plot_combined(chi_err / chi_err.tr(), lbls_list=[['i', 'x', 'y', 'z']] * 2)
plt.savefig("chi_err_matrix.png")
