import numpy as np
import qutip as qt
import time
from rus_trion.evolve import compute_chois_pnr
from rus_trion.config import ZPG, Timing, CyclicInterferometer, Trion

trion = Trion()
zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
timing = Timing(t_initial=0.0, t_final=7.0, steps=1000)
targets = compute_chois_pnr(trion, zpg, timing)

error = []
for n in range(2, 20):
    trion = Trion()
    zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
    timing = Timing(t_initial=0.0, t_final=7.0, steps=n)
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
    error.append((t1 / 81, err / ct))

print(error)
