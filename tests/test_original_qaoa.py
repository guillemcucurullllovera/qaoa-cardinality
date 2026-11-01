"""Retained numerical statevector/mixer tests; exclusions are explicit in TEST_SCOPE.json."""

import numpy as np
import pytest
import time

from qaoa_cardinality.qaoa_v2 import (
    QAOAv2, QAOAConfig, build_cost_hamiltonian,
    qaoa_statevector, energy_expectation, plus_state,
)
from qaoa_cardinality.mixers import (
    correlation_xy_pairs, ring_xy_pairs, apply_xy_pair, initial_dicke_state,
)
from qaoa_cardinality.sk_warmstart import sk_warmstart_angles


def make_sk_qubo(n: int, seed: int) -> np.ndarray:
    """SK spin-glass QUBO: Gaussian couplings, zero diagonal."""
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n, n)) / np.sqrt(n)
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return J


def make_correlated_qubo(n: int, seed: int) -> tuple:
    """
    Financial-style QUBO with block correlation structure.
    Returns (J, corr_matrix) where J = corr_matrix.
    """
    rng = np.random.default_rng(seed)
    # 2 asset clusters: within-cluster high correlation, between low
    half = n // 2
    corr = np.eye(n) * 0.5
    for i in range(half):
        for j in range(half):
            if i != j:
                corr[i, j] = 0.7 + 0.2 * rng.standard_normal()
    for i in range(half, n):
        for j in range(half, n):
            if i != j:
                corr[i, j] = 0.7 + 0.2 * rng.standard_normal()
    # symmetrize and normalize
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 0)
    J = corr.copy()
    return J, corr


# ── Core operation tests ──────────────────────────────────────────────────────

class TestCostHamiltonian:
    def test_shape(self):
        J = make_sk_qubo(4, 0)
        H = build_cost_hamiltonian(J)
        assert H.shape == (2**4,), f"Expected (16,), got {H.shape}"

    def test_real_valued(self):
        J = make_sk_qubo(4, 0)
        H = build_cost_hamiltonian(J)
        assert np.isreal(H).all(), "Cost Hamiltonian must be real-valued."

    def test_symmetric_qubo_symmetric_energy(self):
        """All-equal J_ij = c: energy must be same for bitstrings with same Hamming weight."""
        n = 4
        c = 1.0
        J = c * np.ones((n, n))
        np.fill_diagonal(J, 0)
        H = build_cost_hamiltonian(J)
        # Hamming weight 2 states: 0011, 0101, 0110, 1001, 1010, 1100
        hw2 = [k for k in range(2**n) if bin(k).count('1') == 2]
        energies = H[hw2]
        np.testing.assert_allclose(
            energies, energies[0],
            atol=1e-10, err_msg="All HW=2 states must have equal energy for uniform J."
        )


class TestXYMixer:
    def test_xy_pair_preserves_norm(self):
        rng = np.random.default_rng(0)
        state = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        state /= np.linalg.norm(state)
        new_state = apply_xy_pair(state, 0, 1, beta=0.3)
        np.testing.assert_allclose(
            np.linalg.norm(new_state), 1.0, atol=1e-12,
            err_msg="XY pair must preserve norm."
        )

    def test_xy_pair_preserves_hamming_weight(self):
        """XY mixer swaps |01⟩↔|10⟩: Hamming weight is preserved."""
        # Start in |01⟩ (qubit 0=1, qubit 1=0) = basis state 1
        state = np.zeros(4, dtype=complex)
        state[1] = 1.0  # |01⟩ in 2-qubit space
        new_state = apply_xy_pair(state, 0, 1, beta=np.pi / 4)
        # Should mix |01⟩ and |10⟩ (basis states 1 and 2) only
        assert abs(new_state[0]) < 1e-12, "|00⟩ must be unaffected"
        assert abs(new_state[3]) < 1e-12, "|11⟩ must be unaffected"
        assert abs(new_state[1]) ** 2 + abs(new_state[2]) ** 2 > 0.99

    def test_dicke_state_normalized(self):
        state = initial_dicke_state(4, 2)
        np.testing.assert_allclose(np.linalg.norm(state), 1.0, atol=1e-12)

    def test_dicke_state_hamming_weight(self):
        n, k = 5, 3
        state = initial_dicke_state(n, k)
        probs = np.abs(state) ** 2
        for idx in range(2**n):
            hw = bin(idx).count('1')
            if hw != k:
                assert abs(probs[idx]) < 1e-12, f"State {idx} (HW={hw}) should have zero amplitude"

    def test_correlation_pairs_more_edges_than_ring(self):
        """Correlation-XY should have ≥ ring-XY edges for a correlated matrix."""
        n = 6
        rng = np.random.default_rng(0)
        corr = rng.uniform(0.4, 0.9, (n, n))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 0)
        ring = ring_xy_pairs(n)
        corr_pairs = correlation_xy_pairs(corr, threshold=0.3)
        assert len(corr_pairs) >= len(ring), (
            f"Correlation-XY ({len(corr_pairs)}) should have ≥ ring ({len(ring)}) edges."
        )


# ── Full QAOA optimization tests ─────────────────────────────────────────────

class TestQAOAv2Optimization:
    def test_energy_finite_after_fit(self):
        n = 6
        J = make_sk_qubo(n, 42)
        qaoa = QAOAv2(QAOAConfig(p=1, n_iter=20, seed=42))
        qaoa.fit(J)
        assert np.isfinite(qaoa.best_energy_), "QAOA energy must be finite after fit."

    def test_approximation_ratio_positive(self):
        """QAOA should at least find a positive-energy solution."""
        n = 6
        J, corr = make_correlated_qubo(n, 0)
        qaoa = QAOAv2(QAOAConfig(p=2, n_iter=50, seed=7))
        qaoa.fit(J, corr_matrix=corr)
        ratio = qaoa.approximation_ratio()
        assert ratio >= 0, f"Approximation ratio must be ≥ 0, got {ratio:.3f}"

    def test_portfolio_weights_shape(self):
        n = 6
        J = make_sk_qubo(n, 0)
        qaoa = QAOAv2(QAOAConfig(p=1, n_iter=20, seed=0))
        qaoa.fit(J)
        w = qaoa.portfolio_weights()
        assert w.shape == (n,), f"Expected shape ({n},), got {w.shape}"

    def test_portfolio_weights_in_range(self):
        n = 6
        J = make_sk_qubo(n, 0)
        qaoa = QAOAv2(QAOAConfig(p=1, n_iter=20, seed=0))
        qaoa.fit(J)
        w = qaoa.portfolio_weights()
        assert (w >= 0).all() and (w <= 1).all(), (
            f"Weights must be in [0,1], got min={w.min():.3f} max={w.max():.3f}"
        )

    def test_sample_portfolio_valid_bits(self):
        n = 6
        J = make_sk_qubo(n, 0)
        qaoa = QAOAv2(QAOAConfig(p=1, n_iter=20, seed=0))
        qaoa.fit(J)
        portfolio = qaoa.sample_portfolio(n_samples=100)
        assert portfolio.shape == (n,)
        assert set(portfolio.tolist()).issubset({0, 1})



    @pytest.mark.parametrize("p", [1, 2])
    def test_runtime_small_n(self, p):
        """Optimization for n=8, p=1/2 should complete quickly."""
        n = 8
        J = make_sk_qubo(n, 0)
        start = time.time()
        qaoa = QAOAv2(QAOAConfig(p=p, n_iter=30, seed=0))
        qaoa.fit(J)
        elapsed = time.time() - start
        assert elapsed < 30.0, f"QAOA n={n}, p={p} took {elapsed:.1f}s > 30s"
