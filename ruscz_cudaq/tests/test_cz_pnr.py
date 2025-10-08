import numpy as np
from qutip import to_super, basis, tensor, concurrence, Qobj

DEF_RTOL = 1e-6
DEF_ATOL = 1e-9


def close(a, b, rtol=DEF_RTOL, atol=DEF_ATOL):
    return np.allclose(a, b, rtol=rtol, atol=atol)


def leq(a, b, tol=1e-12):
    return np.less_equal(a, b + tol)


def geq(a, b, tol=1e-12):
    return np.greater_equal(a + tol, b)


def test_probabilities_normalized(chois_pnr):
    it = np.ndindex(chois_pnr.shape[:4])
    p_sum = sum(float(np.trace(chois_pnr[idx]).real) for idx in it)
    assert close(p_sum, 1.0)


def test_each_choi_is_psd(chois_pnr):
    samples = [(0, (0, 0, 0, 0)), (0.125, (0, 2, 0, 0)), (0.125, (0, 1, 1, 0)),
               (0, (1, 1, 0, 0)), (0, (2, 0, 2, 0))]
    for pr, idx in samples:
        J = chois_pnr[idx]
        w = np.linalg.eigvals(J)
        assert geq(w.min(), 0)
        print(pr, np.trace(J))
        assert close(np.trace(J), pr)


def test_outcome_1010_entangles_plus_plus(chois_pnr):
    """
    Ensure that the (1,0,1,0) outcome map produces an entangled
    two-qubit state when applied to |++><++|.
    """
    choi = chois_pnr[1, 0, 1, 0]
    choi = Qobj(choi, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], superrep='choi')
    assert close(choi.tr(), 0.125)

    supo = to_super(4 * choi)

    # Input state |++><++|
    plus = (basis(2, 0) + basis(2, 1)).unit()
    rho_in = tensor(plus, plus) * tensor(plus, plus).dag()

    rho_out = supo(rho_in)
    p = rho_out.tr()
    assert close(p, 0.125)

    c = concurrence(rho_out / p)
    assert close(c, 1)

    s = 1/32
    phase = np.array([
        [1, +1j, -1j, -1],
        [-1j, 1, -1, +1j],
        [+1j, -1, 1, -1j],
        [-1, -1j, +1j, 1],
    ], dtype=np.complex128)

    assert close(s * phase, np.array(rho_out, dtype=np.complex128))
