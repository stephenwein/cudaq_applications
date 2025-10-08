import numpy as np
import qutip as qt
from itertools import product
import cProfile, pstats, io
from rus_trion.evolve import compute_chois_pnr
from rus_trion.config import ZPG, Timing, CyclicInterferometer, Trion
import time
from cudaq import RungeKuttaIntegrator

trion = Trion(decay=1.0,
              B_static=0.0,
              g_factor=0.2,
              oh_sigma=0.2,
              n_samples=100)
zpg = ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2)
timing = Timing(t_initial=0.0, t_final=5.0)

pr = cProfile.Profile()
pr.enable()
t0 = time.time()
Jn = compute_chois_pnr(trion, zpg, timing, integrator=RungeKuttaIntegrator(order=2, max_step_size=1))
print(time.time() - t0)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
ps.print_stats(10)
print(s.getvalue())

# Probabilities per outcome = Tr(J); should sum to ~1
p_sum = 0.0
for idx in product(range(Jn.shape[0]), range(Jn.shape[1]), range(Jn.shape[2]), range(Jn.shape[3])):
    J = Jn[idx]
    p = float(np.trace(J).real)
    print(idx, "→ p =", f"{p:.6f}")
    p_sum += p
print("sum p =", p_sum)

# Example action on |++⟩ using a particular outcome (convert Choi→super)
plus = (qt.fock(2, 0) + qt.fock(2, 1)).unit()
rho_pp = qt.tensor(plus, plus) * qt.tensor(plus, plus).dag()
# pick some resolved outcome, e.g. k=(1,0,1,0)
E = qt.to_super(4 * qt.Qobj(Jn[1, 0, 1, 0], dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi'))
rho_out = E(rho_pp)
print(rho_out)
print("Tr(out) =", rho_out.tr())
