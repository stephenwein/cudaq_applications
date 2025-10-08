from dataclasses import dataclass, field
import numpy as np
from numpy import array, complex128


@dataclass
class Timing:
    t_initial: float = 0.0
    t_final: float = 10.0
    steps: int = 10

@dataclass
class ZPG:
    n_emitters: int = 2
    # number-resolved truncation (eta_k = 1 - exp(-2πik/(n_trunc + 1)))
    n_trunc: int = 2
    # unitary mixing U over modes (4x4). default: identity
    unitary: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.complex128))

@dataclass
class Trion:
    decay: float = 1.0
    B_static: float = 0.0
    g_factor: float = 0.2
    oh_sigma: float = 0.0
    n_samples: int = 1
    seed: int = 1234

class CyclicInterferometer:
    """hardcoded for now"""

    UNITARY_2 = array([[1, 1, 1, 1],
                       [1, 1, -1, -1],
                       [1, -1, -1j, 1j],
                       [1, -1, 1j, -1j]], dtype=complex128) / 2

    UNITARY_3 = array([[1j, -1j, 1, 1, 0, 0],
                       [1j, -1j, -1, -1, 0, 0],
                       [1, 1, 0, 0, 1, -1],
                       [1, 1, 0, 0, -1, 1],
                       [0, 0, 1, -1, 1, 1],
                       [0, 0, 1, -1, -1, -1]], dtype=complex128) / 2

    UNITARY_4 = array([[1, 1, 0, 0, 0, 0, 1, -1],
                       [1j, -1j, 1, 1, 0, 0, 0, 0],
                       [1j, -1j, -1, -1, 0, 0, 0, 0],
                       [0, 0, 1, -1, 1, 1, 0, 0],
                       [0, 0, 1, -1, -1, -1, 0, 0],
                       [0, 0, 0, 0, 1, -1, 1, 1],
                       [0, 0, 0, 0, 1, -1, -1, -1],
                       [1, 1, 0, 0, 0, 0, -1, 1]], dtype=complex128) / 2

    UNITARY_LIST = [UNITARY_2, UNITARY_3, UNITARY_4]

    def __init__(self, n_modes):
        assert n_modes in [2, 3, 4], "Not Implemented"
        self.unitary = self.UNITARY_LIST[n_modes - 2]

    def compute_unitary(self):
        return self.unitary
