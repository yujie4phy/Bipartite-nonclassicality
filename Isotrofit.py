import numpy as np
from scipy.optimize import minimize
from scipy.linalg import sqrtm

def Optimalr(rho,r_val):
    """
    Compute the optimal r and the maximal fidelity of a given density matrix with an isotropic state.

    Parameters:
    - rho: 4x4 density matrix (numpy array).

    Returns:
    - r_opt: Optimal mixing parameter r that maximizes the fidelity.
    - max_fidelity: The maximal fidelity between rho and the isotropic state.
    """
    # Define Bell state |Φ+⟩ and components
    Phi_plus = np.array([1, 0, 0, 1]) / np.sqrt(2)
    Phi_plus_dm = np.outer(Phi_plus, Phi_plus.conj())  # |Φ+⟩⟨Φ+|
    I_d2 = np.eye(4) / 4  # Scaled identity matrix

    # Define the isotropic state function
    def isotropic_state(r):
        return r * Phi_plus_dm + (1 - r) * I_d2

    # Define the fidelity function based on the correct definition
    def fidelity(r):
        iso_state = isotropic_state(r)  # Compute ρ_iso(r)
        sqrt_rho = sqrtm(rho)  # Compute the square root of ρ
        inner_product = sqrtm(sqrt_rho @ iso_state @ sqrt_rho)  # Compute sqrt(√ρ ρ_iso √ρ)
        return np.abs(np.trace(inner_product))**2  # Fidelity

    # Objective function to minimize (negative fidelity for maximization)
    def fidelity_to_minimize(r):
        return -fidelity(r)

    # Perform the optimization
    initial_guess = 0.5
    bounds = [(0, 1)]  # r must be between 0 and 1
    result = minimize(fidelity_to_minimize, x0=initial_guess, bounds=bounds)

    # Extract optimal r and maximum fidelity
    r_opt = result.x[0]
    max_fidelity = -result.fun
    fidelity_o =fidelity(r_val)
    return r_opt, max_fidelity,fidelity_o, isotropic_state(r_opt)