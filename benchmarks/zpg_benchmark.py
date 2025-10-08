from zpgenerator import *
import qutip as qt
import numpy as np
import perceval as pcvl
import time
import perceval.components.unitary_components as comp

c = pcvl.Circuit(4)
c.add(0, pcvl.BS.H())
c.add(2, pcvl.BS.H())
c.add([0, 1, 2, 3], comp.PERM([0, 2, 1, 3]))
c.add(3, pcvl.PS(-np.pi/2))
c.add(0, pcvl.BS.H())
c.add(2, pcvl.BS.H())
pcvl.pdisplay(c)

c_rus = Circuit.from_perceval(c)

trion = Source.trion()
trion.update_default_parameters(parameters={'theta': 0, 'phi': 0, 'theta_c': np.pi / 4,
                                            'phi_c': -np.pi / 2, 'area': np.sqrt(2) * np.pi})

p_rus = Processor() // ([0, 2], trion) // c_rus // (list(range(4)), Detector.pnr(2))
p_rus.final_time = 5

basis = [trion.states['|spin_up>'], trion.states['|spin_down>']]
basis = [qt.tensor(b1, b2) for b1 in basis for b2 in basis]  # take tensor products

targets = p_rus.conditional_channels(basis=basis, select=[1, 3])

tvd_list = []
for n in range(1, 21, 5):
    t0 = time.time()
    channels = p_rus.conditional_channels(basis=basis, select=[1, 3],
                                              options=qt.Options(atol=0.1 / n**4, rtol=1 / n**4))
    t1 = time.time() - t0
    tvd = 0
    for idx, targ in targets.items():
       tvd += abs(targ.tr() - channels[idx].tr()) / 2
    tvd_list.append((t1 / 81, tvd))
