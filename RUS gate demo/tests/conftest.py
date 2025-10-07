import pytest
from rus_trion.evolve import compute_chois_pnr
from rus_trion.config import ZPG, Timing, CyclicInterferometer, Trion


@pytest.fixture(scope="session")
def chois_pnr():
    return compute_chois_pnr(
        Trion(),
        ZPG(n_emitters=2, unitary=CyclicInterferometer(2).compute_unitary(), n_trunc=2),
        Timing(t_initial=0, t_final=30.0, steps=60))
