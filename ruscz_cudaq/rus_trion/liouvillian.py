import numpy as np
from cudaq import boson, SuperOperator
from itertools import product
from .config import ZPG, Trion


# Pauli operators using cudaq.boson (sparse)
def sx(i):  # σ_x
    return boson.create(i) + boson.annihilate(i)

def sy(i):  # σ_y
    return 1j * (boson.create(i) - boson.annihilate(i))

def sz(i):  # σ_z
    return boson.identity(i) - 2.0 * boson.number(i)

# convert Hamiltonian to Liouvillian
def liouvillian_from_hamiltonian(H):
    """
    Return the superoperator L_H implementing -i[H,·]:
      L_H(ρ) = -i H ρ + i ρ H
    """
    LH = SuperOperator.left_multiply(-1j * H)
    LH += SuperOperator.right_multiply(1j * H)
    return LH

# ---------- Hamiltonian builder ----------
def _sample_spin_hamiltonian(spins, trion: Trion, rng):
    """
    Construct H = sum_i g * (B_i · σ_i) where B_i = B_static * δ_iy + δB_i.
    δB_i is a quasi-static Overhauser field drawn once per emitter.
    """
    spins = list(spins)
    n = len(spins)
    if n == 0:
        raise ValueError("spins is empty; cannot build a Hamiltonian with no spins.")

    # Normalize oh_sigma into an array of shape (n,3)
    oh_sigma = trion.oh_sigma
    sig = np.array([oh_sigma, oh_sigma, oh_sigma], dtype=float)
    oh_sigmas = np.tile(sig, (n, 1))

    # Sample quasi-static δB_i
    dB = rng.normal(loc=0.0, scale=oh_sigmas)  # shape (n,3)

    # Add static field: B_y = B_static + δB_y
    B_totals = dB.copy()
    B_totals[:, 1] += float(trion.B_static)

    # Build H = sum_i g * (B_i,x σ_x + B_i,y σ_y + B_i,z σ_z)
    H = 0.0 * boson.identity(spins[0])  # typed zero in boson operator family
    for i, (bx, by, bz) in zip(spins, B_totals):
        H += (trion.g_factor / 2) * (bx * sx(i) + by * sy(i) + bz * sz(i))

    return liouvillian_from_hamiltonian(H)

def dissipator(C, Cd):
    L = SuperOperator.left_right_multiply(C, Cd)
    CC = Cd * C
    L += SuperOperator.left_multiply(-0.5 * CC)
    L += SuperOperator.right_multiply(-0.5 * CC)
    return L

def lincomb(coeffs, ops, seed_idx=0):
    """Typed linear combination."""
    out = 0.0 * boson.identity(seed_idx)
    for c, o in zip(coeffs, ops):
        if c != 0: out = out + c * o
    return out

def build_liouvillians(trion: Trion, zpg: ZPG):
    # Layout (LSB-first): [aux0..aux{n-1}, opt0, spn0, opt1, spn1, ...]
    n_emitters = zpg.n_emitters
    unitary = zpg.unitary
    n_trunc = zpg.n_trunc

    n_modes   = 2 * n_emitters
    photons   = list(range(n_emitters, 3 * n_emitters, 2))
    spins     = list(range(n_emitters + 1, 3 * n_emitters, 2))

    # Spin projectors & ops in boson(2)
    P0 = [boson.identity(i) - boson.number(i) for i in spins]  # |0><0|
    P1 = [boson.number(i) for i in spins]                      # |1><1|
    a  = boson.annihilate
    ad = boson.create

    # Local, spin-resolved channels in order [R0,L0,R1,L1,...]
    jump_loc   = [op for k,j in enumerate(photons) for op in (P0[k]*a(j), P1[k]*a(j))]
    jump_loc_d = [op for k,j in enumerate(photons) for op in (P0[k]*ad(j), P1[k]*ad(j))]

    # Detection mixing (column mixing: C'_m = Σ_j U[j,m] C_j)
    jump_mix   = [lincomb(unitary[:, m],             jump_loc,   seed_idx=photons[0]) for m in range(n_modes)]
    jump_mix_d = [lincomb(unitary[:, m].conjugate(), jump_loc_d, seed_idx=photons[0]) for m in range(n_modes)]

    # PNR: per-mode η_k grid and cartesian product of 4 modes
    gamma = float(trion.decay)
    eps = 1e-16 + 1e-16j
    ks   = np.arange(n_trunc + 1, dtype=float)
    base = -gamma * (1.0 - np.exp(-2j * np.pi * ks / (n_trunc + 1)) + eps)  # η_k = 1 - e^{+i 2π k/(n+1)}
    per_mode_etas = [base.copy() for _ in range(n_modes)]
    eta_configs = list(product(*per_mode_etas))  # tuples (η_Ru, η_Rd, η_Lu, η_Ld)
    eta_configs = [eta for eta in eta_configs]

    # Sample quasi-static Hamiltonians (in Liouvillian representation)
    LH_list = []
    rng = np.random.default_rng(trion.seed)
    for _ in range(trion.n_samples):
        LH_list.append(_sample_spin_hamiltonian(spins, trion, rng))

    L_emission = SuperOperator.left_multiply(0.0 * boson.identity(photons[0]))
    for Cj, Cdj in zip(jump_loc, jump_loc_d):
        L_emission += dissipator(gamma * Cj, Cdj)

    # Build batched Liouvillians: Σ_j D[C_j] + Σ_m ( -η_m ) J[C'_m]
    L_list = []
    for LH in LH_list:
        for etas in eta_configs:
            L = SuperOperator.left_multiply((0.0 + 0.0j) * boson.identity(photons[0]))
            L += LH
            L += L_emission
            for Cm, Cdm, eta in zip(jump_mix, jump_mix_d, etas):
                L += SuperOperator.left_right_multiply(eta * Cm, Cdm)
            L_list.append(L)

    return L_list