import numpy as np
import qutip as qt
import time
from rus_trion.evolve import compute_chois_pnr
from rus_trion.config import ZPG, Timing, CyclicInterferometer, Trion
from cudaq import RungeKuttaIntegrator

trion = Trion(oh_sigma=0.1, n_samples=100)
zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
timing = Timing(sequence=[0, 5])
targets = compute_chois_pnr(trion, zpg, timing, integrator=RungeKuttaIntegrator(order=4, max_step_size=0.5))

error = []
for n in range(1, 21):
    t0 = time.time()
    chois = compute_chois_pnr(trion, zpg, timing, RungeKuttaIntegrator(order=2, max_step_size=1))
    t1 = time.time() - t0

    err = 0
    for idx in np.ndindex(targets.shape[:4]):
        E_targ = qt.Qobj(targets[idx], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
        E = qt.Qobj(chois[idx], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
        err += abs(E.tr() - E_targ.tr()) / 2
    error.append((t1 / 81, err))

print(error)
