import numpy as np
import cvxpy as cp


def partial_transpose(A):
    # Explicitly specify order='C' for reshaping operations
    A_reshaped = cp.reshape(A, (2, 2, 2, 2), order='C')
    A_pt_reshaped = cp.transpose(A_reshaped, axes=(0, 3, 2, 1))
    return cp.reshape(A_pt_reshaped, (4, 4), order='C')

def haar_random_unitary(n):
    Z = (np.random.randn(n, n) + 1j * np.random.randn(n, n)) / np.sqrt(2.0)
    Q, R = np.linalg.qr(Z)
    d = np.diag(R)
    ph = d / np.abs(d)
    Q = Q * ph
    return Q
def HaarEPR():
    dim = 2
    U = haar_random_unitary(dim)
    phi_plus = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
    I = np.eye(dim, dtype=complex)
    I_tensor_U = np.kron(I, U)
    psi = I_tensor_U @ phi_plus
    return psi
def bell_state(n):
    Bellsets=[np.array([1, 0, 0, 1]) / np.sqrt(2),np.array([1, 0, 0, -1]) / np.sqrt(2),
 np.array([0, 1, 1, 0]) / np.sqrt(2), np.array([0, 1, -1, 0]) / np.sqrt(2)]
    for i in range(n):
        Bellsets.append(HaarEPR())
    return Bellsets

def Isotropic(psi):
    proj = np.outer(psi, psi.conjugate())
    I4 = np.eye(4, dtype=complex)
    return 0.5 * proj + 0.125 * I4


def Udecomposition(sigma, n=20):
    """
    Solve the SDP:
      Maximize p[0]
      subject to:
        sum_i p[i]*rhos[i] + X = sigma
        sum_i p[i] + trace(X) = 1
        p[i] >= 0
        X >= 0, partial_transpose(X) >= 0

    Returns: (feasible_flag, p_vals, X_val, max_p0)
      - feasible_flag: bool
      - p_vals: array of p[i]
      - X_val: the 4x4 "unnormalized" separable component
      - max_p0: the solver's maximum value of p[0]
    """
    n = 20
    EPRsets = bell_state(n)
    rhos = [Isotropic(EPR) for EPR in EPRsets]
    # Define a CVXPY variable for the coefficients p (each p_i >= 0)
    p = cp.Variable(n, nonneg=True)


    sum_rho = 0
    for i in range(n):
        sum_rho += p[i] * rhos[i]

    # Define the separable remainder.
    SEP = sigma - sum_rho

    # Constraint 1: SEP must be positive semidefinite.
    constraints = [SEP >> 0]

    # Constraint 2: The partial transpose of SEP (with respect to the second qubit) must also be PSD.
    sigma_pt = partial_transpose(sigma)
    sum_rho_pt = 0
    for i in range(n):
        rho_i_pt = partial_transpose(rhos[i])
        sum_rho_pt += p[i] * rho_i_pt
    SEP_pt = sigma_pt - sum_rho_pt
    constraints.append(SEP_pt >> 0)

    # Define the objective: maximize the total weight of the given states, sum_i p_i.
    objective = cp.Maximize(cp.sum(p))

    # Set up and solve the SDP.
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS)  # You may switch to another SDP solver if available.

    # Check if the problem was solved successfully.
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        print("The SDP failed to find an optimal solution. Problem status:", prob.status)
        return None, None

    print("Decomposition found")

    # Construct the separable remainder using the optimal p values.
    p_opt = p.value
    SEP_opt = sigma.copy()
    for i in range(n):
        SEP_opt -= p_opt[i] * rhos[i]

    return p_opt, SEP_opt

