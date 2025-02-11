import numpy as np
from gurobipy import Model, GRB, QuadExpr
import Bases
from scipy.optimize import minimize
# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
# Pauli basis for a single qubit
pauli_basis = [I, X, Y, Z]
def GPTstate(rho):
    G = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            G[i, j] = np.trace(rho @ np.kron(pauli_basis[i], pauli_basis[j]))
    return np.real(G)
def GPTPOVM(povm):
    pauli_vectors = []
    for effect in povm:
        n = np.zeros(4, dtype=complex)
        for i in range(4):
            # Compute the coefficient for each Pauli matrix component
            n[i] = 0.5 * np.trace(effect @ pauli_basis[i])
        pauli_vectors.append(n)
    return np.real(np.array(pauli_vectors))
def Cal_P(G, n, m):
    calP = np.zeros((len(n), len(m)))
    for i in range(len(n)):
        for j in range(len(m)):
            # Calculate P_ij = n_i^T G m_j
            calP[i, j] = n[i].T @ G @ m[j]
    return calP
# Optimization functions

def optimize_G(n, m, P, Pu, G):
    """Optimize G given fixed n and m using SciPy, with G[0][0] fixed to 1."""
    # Flatten initial_G excluding G[0][0]
    initial_guess = np.delete(G.flatten(), 0)
    def objective(G_flat):
        G = np.zeros((4, 4))
        G[0, 0] = 1  # Fix G[0][0] to 1
        G_flat_full = np.insert(G_flat, 0, 1)  # Rebuild G with G[0][0] = 1
        G = G_flat_full.reshape(4, 4)
        error = 0.0
        for i in range(len(n)):
            for j in range(len(m)):
                tilde_P_ij = np.dot(n[i], G @ m[j])
                error += ((P[i, j] - tilde_P_ij) / (Pu[i, j])) ** 2
        return error
    result = minimize(objective, initial_guess, method='L-BFGS-B')
    if result.success:
        G_optimized = np.zeros((4, 4))
        G_optimized[0, 0] = 1  # Fix G[0][0] to 1
        G_flat_full = np.insert(result.x, 0, 1)
        G_optimized = G_flat_full.reshape(4, 4)
        print(f"Optimized Objective Value: {result.fun}")
        return G_optimized
    else:
        print("Optimization failed")
        return initial_guess


def optimize_n(G, m, P,Pu, n,pair1):
    """Optimize n given fixed G and m using SciPy with direct constraints."""
    initial_guess = np.array(n).flatten()
    num_n=len(n)
    num_m=len(m)
    def objective(n_flat):
        n = n_flat.reshape(num_n, 4)
        error = 0.0
        for i in range(num_n):
            for j in range(num_m):
                tilde_P_ij = np.dot(n[i], G @ m[j])
                error += ((P[i, j] - tilde_P_ij) / (Pu[i, j])) ** 2
        return error
    def norm_cons(n_flat):
        n = n_flat.reshape(num_n, 4)
        constraints = []
        for idx_1, idx_2 in pair1:
            constraints.extend((n[idx_1] + n[idx_2] - np.array([1,0,0,0])).tolist())
        return constraints
    constraints = {'type': 'eq', 'fun': norm_cons}
    result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
    if result.success:
        n_optimized = result.x.reshape(num_n, 4)
        print(f"Optimized Objective Value for n: {result.fun}")
        return n_optimized
    else:
        print("Optimization for n failed")
        return initial_guess

def optimize_m(G, n, P, Pu, m,pair2):
    """Optimize m given fixed G and n using SciPy with direct constraints."""
    initial_guess = np.array(m).flatten()
    num_n=len(n)
    num_m=len(m)
    def objective(m_flat):
        m = m_flat.reshape(num_m, 4)
        error = 0.0
        for i in range(num_n):
            for j in range(num_m):
                tilde_P_ij = np.dot(n[i], G @ m[j])
                error += ((P[i, j] - tilde_P_ij) / (Pu[i, j])) ** 2
        return error
    def pairwise_constraint(m_flat):
        m = m_flat.reshape(num_m, 4)
        constraints = []
        for idx_1, idx_2 in pair2:
            constraints.extend((m[idx_1] + m[idx_2] - np.array([1,0,0,0])).tolist())
        return constraints
    constraints = {'type': 'eq', 'fun': pairwise_constraint}
    result = minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
    if result.success:
        m_optimized = result.x.reshape(num_m, 4)
        print(f"Optimized Objective Value for m: {result.fun}")
        return m_optimized
    else:
        print("Optimization for m failed")
        return initial_guess
def errP(Pnew,P,Pu):
    num_N, num_M = Pnew.shape
    error=0
    for j in range(num_M):
        for i in range(num_N):
            error+= ((P[i, j] - Pnew[i,j]) / Pu[i, j]) ** 2
    return error
tolerance = 1e-3
max_iters = 100
def FitGPT(P,Pu,rho,N,M,pair1,pair2, max_iters=100,tolerance=1e-2):
    G = GPTstate(rho)
    m = GPTPOVM(M)
    n = GPTPOVM(N)
    for iteration in range(max_iters):
        Pold = Cal_P(G, n, m)
        err_old=errP(Pold,P,Pu)
        G = optimize_G(n, m, P, Pu, G)  # Step 1: Optimize G given n and m
        n = optimize_n(G, m, P, Pu, n,pair1)  # Step 2: Optimize n given G and m
        m = optimize_m(G, n, P, Pu, m,pair2)  # Step 3: Optimize m given G and n
        Pnew=Cal_P(G,n,m)
        if abs(errP(Pnew,P,Pu) - err_old) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    #Pnew=SecP(P1,N,M)
    return Pnew
