#!/usr/bin/env python3
"""
Two-qubit state tomography via 36 measurements:
- Linear inversion (as a baseline)
- Maximum Likelihood Estimation (MLE) using the RρR iterative algorithm

Measurements are performed in the eigenbases of the Pauli X, Y, and Z operators.
Each qubit is measured in 3 bases, giving 9 measurement settings.
Each setting provides 4 outcomes, so 9×4=36 measurement outcomes.
"""

import numpy as np


def single_qubit_projectors():
    """
    Returns the three measurement bases (X, Y, Z) for a single qubit.
    Each basis is represented by two projectors.
    """
    # Computational basis states
    ket0 = np.array([[1], [0]], dtype=complex)
    ket1 = np.array([[0], [1]], dtype=complex)

    # Z basis: |0><0|, |1><1|
    Pz0 = ket0 @ ket0.conj().T
    Pz1 = ket1 @ ket1.conj().T

    # X basis: |+> = (|0>+|1>)/sqrt(2), |-> = (|0>-|1>)/sqrt(2)
    ket_plus = (ket0 + ket1) / np.sqrt(2)
    ket_minus = (ket0 - ket1) / np.sqrt(2)
    Px0 = ket_plus @ ket_plus.conj().T
    Px1 = ket_minus @ ket_minus.conj().T

    # Y basis: |+i> = (|0>+i|1>)/sqrt(2), |-i> = (|0>-i|1>)/sqrt(2)
    ket_plus_i = (ket0 + 1j * ket1) / np.sqrt(2)
    ket_minus_i = (ket0 - 1j * ket1) / np.sqrt(2)
    Py0 = ket_plus_i @ ket_plus_i.conj().T
    Py1 = ket_minus_i @ ket_minus_i.conj().T

    return {
        'X': [Px0, Px1],
        'Y': [Py0, Py1],
        'Z': [Pz0, Pz1]
    }


def generate_measurement_operators():
    """
    Generates the list of 36 two-qubit measurement operators as tensor products
    of single-qubit projectors. Also returns a list of labels (basis settings).
    """
    single_ops = single_qubit_projectors()
    meas_ops = []  # List to hold 4x4 measurement operators
    settings = []  # List of labels for each measurement operator
    bases = ['X', 'Y', 'Z']
    for b1 in bases:
        for b2 in bases:
            # For each measurement setting (e.g., X on qubit1 and Y on qubit2)
            for proj1 in single_ops[b1]:
                for proj2 in single_ops[b2]:
                    M = np.kron(proj1, proj2)
                    meas_ops.append(M)
                    settings.append((b1, b2))
    return meas_ops, settings


def simulate_probabilities(rho, meas_ops):
    """
    Computes the ideal (noise-free) probabilities for each measurement outcome.
    p_k = Tr(M_k rho)
    """
    probs = np.array([np.real(np.trace(rho @ M)) for M in meas_ops])
    return probs


def simulate_counts(rho, meas_ops, shots_per_setting=1000):
    """
    Simulates a tomography experiment.
    For each of the 9 measurement settings (each with 4 outcomes), a multinomial
    experiment with 'shots_per_setting' shots is simulated.
    Returns an array of 36 estimated probabilities (one per measurement outcome).
    """
    full_probs = simulate_probabilities(rho, meas_ops)
    measured_probs = []
    # There are 9 settings (each with 4 outcomes)
    for i in range(0, len(meas_ops), 4):
        p_group = full_probs[i:i + 4]
        # Simulate shot noise for this setting:
        counts = np.random.multinomial(shots_per_setting, p_group)
        # Estimated probabilities for this setting:
        measured_probs.extend(counts / shots_per_setting)
    return np.array(measured_probs)


def reconstruct_density_matrix(measured_probs, meas_ops):
    """
    Reconstructs the density matrix via linear inversion (least–squares).
    Note: Linear inversion can sometimes produce nonphysical density matrices.
    """
    # Build the matrix A where each row is the flattened measurement operator
    A = np.array([M.reshape(-1) for M in meas_ops])
    p = measured_probs  # 36-dimensional vector
    # Solve A r = p for r (vectorized density matrix)
    r, residuals, rank, s = np.linalg.lstsq(A, p, rcond=None)
    rho_est = r.reshape(4, 4)
    # Force Hermiticity
    rho_est = (rho_est + rho_est.conj().T) / 2
    # Normalize trace to 1
    rho_est = rho_est / np.trace(rho_est)
    return rho_est


def max_likelihood_tomography(measured_probs, meas_ops, tol=1e-6, max_iter=1000):
    """
    Reconstructs the density matrix using Maximum Likelihood Estimation via the
    iterative RρR algorithm. This algorithm ensures the reconstructed density
    matrix is physical (i.e., positive semidefinite with trace 1).

    The iterative update is:
      1. Compute p_k = Tr(M_k ρ) for each outcome.
      2. Form the operator R(ρ) = sum_k (f_k/p_k) M_k, where f_k are the measured
         probabilities for each outcome.
      3. Update ρ_new = [R ρ R] / Tr(R ρ R)
      4. Repeat until convergence.
    """
    d = meas_ops[0].shape[0]  # Dimension (should be 4 for two qubits)
    rho = np.eye(d) / d  # Start with the maximally mixed state

    for iteration in range(max_iter):
        # Compute theoretical probabilities for each outcome given current ρ
        p = np.array([np.trace(M @ rho) for M in meas_ops]).real
        # Avoid division by zero:
        p = np.clip(p, 1e-10, None)
        # Construct the operator R(ρ)
        R = np.zeros_like(rho, dtype=complex)
        for k in range(len(meas_ops)):
            R += (measured_probs[k] / p[k]) * meas_ops[k]
        # Update ρ using the RρR formula
        rho_new = R @ rho @ R
        rho_new = rho_new / np.trace(rho_new)
        # Check for convergence using the Frobenius norm
        diff = np.linalg.norm(rho_new - rho, ord='fro')
        rho = rho_new
        if diff < tol:
            print("MLE converged after", iteration, "iterations")
            break
    return rho


def bell_state_density():
    """
    Returns the density matrix for the Bell state |φ⁺⟩ = (|00⟩ + |11⟩)/√2.
    """
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    return rho


def main():
    # Generate the 36 measurement operators
    meas_ops, settings = generate_measurement_operators()

    # Define the true state (Bell state)
    rho_true = bell_state_density()
    print("True density matrix:\n", rho_true)

    # Simulate an experiment:
    shots_per_setting = 1000
    measured_probs = simulate_counts(rho_true, meas_ops, shots_per_setting=shots_per_setting)

    # Reconstruct using linear inversion
    rho_lin = reconstruct_density_matrix(measured_probs, meas_ops)
    print("\nReconstructed density matrix via linear inversion:\n", rho_lin)

    # Reconstruct using Maximum Likelihood Estimation (MLE)
    rho_ml = max_likelihood_tomography(measured_probs, meas_ops, tol=1e-6, max_iter=1000)
    print("\nReconstructed density matrix via Maximum Likelihood Estimation (MLE):\n", rho_ml)


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    main()