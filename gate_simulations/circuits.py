from numpy import array, complex128


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
