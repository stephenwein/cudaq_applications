import numpy as np
import qutip as qt
import time
import tqdm
from rus_trion.evolve import compute_chois_pnr
from rus_trion.config import ZPG, Timing, CyclicInterferometer, Trion


## Compute target matrices
trion = Trion()
zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
timing = Timing(t_initial=0.0, t_final=10.0, steps=20)
targets = compute_chois_pnr(trion, zpg, timing)

## Compute noisy matrices
choi_list = []
sigma_list = np.linspace(0, 0.3, 30)

t0 = time.time()
for sigma in tqdm.tqdm(sigma_list):
    trion = Trion(oh_sigma=sigma, n_samples=100)
    zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
    timing = Timing(t_initial=0.0, t_final=5.0)

    choi_list.append(compute_chois_pnr(trion, zpg, timing,integrator=RungeKuttaIntegrator(order=2, max_step_size=1)))
simulation_time = time.time() - t0
print('Simulation time: ', simulation_time)

## Compute Bell state fidelity
plus = (qt.fock(2, 0) + qt.fock(2, 1)).unit()
rho_pp = qt.tensor(plus, plus) * qt.tensor(plus, plus).dag()

fidelity_list = []
efficiency_list = []
success = [(1, 0, 1, 0), (1, 0, 0, 1), (0, 1, 1, 0), (0, 1, 0, 1)]
for chois in choi_list:
    fid = 0
    eff = 0
    for outcome in success:
        E = qt.to_super(4 * qt.Qobj(chois[outcome], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi'))
        E_targ = qt.to_super(4 * qt.Qobj(targets[outcome], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi'))

        rho_out = E(rho_pp)
        rho_targ = E_targ(rho_pp)
        eff += rho_out.tr()
        fid += qt.fidelity(rho_out / rho_out.tr(), rho_targ / rho_targ.tr())**2
    fidelity_list.append(fid / 4)
    efficiency_list.append(eff)

print("Bell-state fidelity: ", fidelity_list)
print("Bell-state efficiency: ", efficiency_list)

## Compute process fidelity
fidelity_dict = {}
for chois in choi_list:
    fid = 0
    ct = 0
    fid_dict = {}
    for idx in np.ndindex(targets.shape[:4]):
        E_targ = qt.Qobj(targets[idx], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
        if abs(E_targ.tr()) > 0.001:
            E = qt.Qobj(chois[idx], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
            fid = qt.process_fidelity(E / E.tr(), E_targ / E_targ.tr())
            if idx in fidelity_dict.keys():
                fidelity_dict[idx].append(abs(fid))
            else:
                fidelity_dict.update({idx: [abs(fid)]})

print("Average process fidelity")
for idx, fids in fidelity_dict.items():
    print(idx, fids)

np.save('choi_matrices.npy', choi_list, allow_pickle=True)
np.save('choi_targets.npy', targets, allow_pickle=True)
