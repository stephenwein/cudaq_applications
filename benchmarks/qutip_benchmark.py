import qutip as qt
import numpy as np
import time
from scipy.integrate import simpson

# This script mimics the computation required to simulate the Choi matrix using standard QRT in QuTiP

r = lambda: np.random.random() * 0.01
H = lambda: r() * qt.sigmax() + r() * qt.sigmay() + r() * qt.sigmaz()
H = qt.tensor(H(), *[qt.qeye(2)]*3) + qt.tensor(*[qt.qeye(2)]*2, H(), qt.qeye(2))

c1 = qt.tensor(qt.num(2), qt.destroy(2), *[qt.qeye(2)]*2)
c2 = qt.tensor(qt.qeye(2) - qt.num(2), qt.destroy(2), *[qt.qeye(2)]*2)
c3 = qt.tensor(*[qt.qeye(2)]*2, qt.num(2), qt.destroy(2))
c4 = qt.tensor(*[qt.qeye(2)]*2, qt.qeye(2) - qt.num(2), qt.destroy(2))
c_ops = [c1, c2, c3, c4]
qft = np.array([[1, 1, 1, 1],
                [1, 1, -1, -1],
                [1, -1, -1j, 1j],
                [1, -1, 1j, -1j]]) / 2
d_ops = [sum(qft[i, j] * c_ops[j] for j in range(4)) for i in range(4)]

istate = (qt.tensor(qt.basis(2, 0), qt.basis(2, 1)) +
          qt.tensor(qt.basis(2, 1), qt.basis(2, 1))) / np.sqrt(2)
istate = qt.tensor(istate, istate)

tvd_list = []
for res in np.linspace(5, 205, 10):
    t0 = time.time()
    tvd = 0
    ct = 0
    for i in range(4):
        for j in range(4):
            if j >= i:
                ct += 1
                res = int(res)
                dat = qt.correlation_3op_2t(H, state0=istate,
                                      tlist=np.linspace(0, 5, res),
                                      taulist=np.linspace(0, 5, res),
                                      c_ops=c_ops,
                                      a_op=d_ops[i].dag(), b_op=d_ops[j].dag() * d_ops[j], c_op=d_ops[i])
                val = (1 if i==j else 2)*simpson(simpson(dat)) * (5/res)**2
                targ = (1 - np.exp(-5))/8 if val > 0.1 else 0
                tvd += abs(val - targ) / 2
    t1 = (time.time() - t0) / ct * 16**2  # 16**2, one for each element of the Choi matrix, one outcome
    tvd_list.append((t1, tvd))
