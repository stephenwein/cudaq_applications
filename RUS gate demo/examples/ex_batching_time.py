import qutip as qt
import numpy as np
import time
import tqdm
from rus_trion.evolve import compute_chois_pnr
from rus_trion.config import ZPG, Timing, CyclicInterferometer, Trion

trion = Trion(oh_sigma=0.1, n_samples=100)
zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
timing = Timing(t_initial=0.0, t_final=7.0, steps=20)
targets = compute_chois_pnr(trion, zpg, timing)

results = []
for n in tqdm.tqdm([6]):
    trion = Trion(oh_sigma=0.1, n_samples=10*n)
    zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
    timing = Timing(t_initial=0.0, t_final=7.0, steps=7)
    t0 = time.time()
    chois = compute_chois_pnr(trion, zpg, timing)
    t1 = time.time() - t0

    err = 0
    ct  = 0
    for idx in np.ndindex(targets.shape[:4]):
        E_targ = qt.Qobj(targets[idx], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
        if abs(E_targ.tr()) > 0.001:
            E = qt.Qobj(chois[idx], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
            diff = abs((E - E_targ).full().reshape(16 * 16))
            sums = (abs(E.full()) + abs(E_targ.full())).reshape(16 * 16)
            err += sum(diff) / sum(sums)
            ct += 1
    results.append((n, t1, err / ct))

print(results)
