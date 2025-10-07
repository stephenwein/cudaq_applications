import numpy as np
import cudaq
from cudaq import Schedule, evolve, SuperOperator
from cupy import asnumpy
from .config import ZPG, Timing, Trion
from .utils import lsb_to_msb, block_auxspin_to_choi, ptrace
from .liouvillian import build_liouvillians
from .states import choi_initial_state


def _evolve_choi(L_list, zpg: ZPG, schedule: Schedule, integrator=None, max_batch_size=None):
    n_emitters = zpg.n_emitters
    n_qubits  = 3 * n_emitters
    n_modes   = 2 * n_emitters
    auxilia   = list(range(n_emitters))
    spins     = list(range(n_emitters + 1, 3 * n_emitters, 2))
    dims_map  = {i: 2 for i in range(n_qubits)}
    dims_list = [2] * (3*n_emitters)
    sizes = [zpg.n_trunc + 1] * n_modes
    total_size = np.prod(sizes)

    # Choi input: optics in |1...1>, (aux_i,spn_i) Bell-tied
    rho0 = choi_initial_state(n_emitters)

    # Evolve batched
    results = evolve(L_list, dimensions=dims_map, schedule=schedule, initial_state=rho0,
                     store_intermediate_results=False,
                     integrator=integrator, max_batch_size=max_batch_size)

    # Average over samples
    d = 2 ** n_qubits
    n_samples = int(len(L_list) / total_size)
    results = np.array([asnumpy(r.final_state()).reshape(d, d) for r in results])
    results = results.reshape(n_samples, total_size, d, d).mean(axis=0)

    # Reduce to Choi blocks (aux+spins kept; regroup to (spins|aux))
    d = 2 ** n_qubits
    blocks = []
    for A in results:
        A = lsb_to_msb(A, n_subs=3*n_emitters)  # reverse global ordering to be consistent with qutip
        A = ptrace(A, dims_list, auxilia + spins)
        A = block_auxspin_to_choi(A, n_emitters)
        blocks.append(A.astype(np.complex128))

    # Stack and invert generating function with an inverse FFT over detector axes
    d = 2 ** (2 * n_emitters)
    choi_grid = np.stack(blocks, axis=0).reshape(*sizes, d, d)
    choi_list = np.fft.ifftn(choi_grid, axes=tuple(range(n_modes)))

    return choi_list


def compute_chois_pnr(trion: Trion, zpg: ZPG, timing: Timing, integrator=None, max_batch_size=None):
    """
    PNR ZPG via generating function:
    """
    cudaq.set_target("dynamics")

    # Time grid
    sched = Schedule(np.linspace(timing.t_initial, timing.t_final, timing.steps), ["t"])

    # Generate list of tilted Liouvillians
    L_list = build_liouvillians(trion, zpg)

    # Evolve batched choi
    choi_list = _evolve_choi(L_list, zpg, sched, integrator, max_batch_size)

    return choi_list