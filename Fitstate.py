import cvxpy as cp
import numpy as np
import qutip as qt
import scipy
def optimize_rho(N, M, P, Pu,rho):
    num_N = len(N)
    num_M = len(M)
    dim_rho = len(rho)  # Dimension of N_i and M_j (2x2)
    """Optimize rho given fixed N and M."""
    rho = cp.Variable((dim_rho, dim_rho), hermitian=True)
    objective = 0
    # Build the objective function
    for i in range(num_N):
        for j in range(num_M):
            Ni_otimes_Mj = np.kron(N[i], M[j])
            Ni_otimes_Mj_cvx = cp.Constant(Ni_otimes_Mj) # Convert Ni_otimes_Mj to a cvxpy constant to use it in the objective
            objective += ((P[i, j] - cp.real(cp.trace(Ni_otimes_Mj_cvx @ rho))) / Pu[i, j]) ** 2
    objective = cp.Minimize(objective)
    # Define constraints
    constraints = [ cp.trace(rho) == 1,  # Trace constraint for a density matrix
        rho >> 0]  # Positive semidefinite constraint
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True, solver=cp.MOSEK)
    if prob.status not in ["infeasible", "unbounded"]:
        return rho.value
    else:
        print("Problem is infeasible or unbounded.")
        return None

# Computing chi-error for probability
def errP(Pnew,P,Pu):
    num_N, num_M = Pnew.shape
    error=0
    for j in range(num_M):
        for i in range(num_N):
            error+= ((P[i, j] - Pnew[i,j]) / Pu[i, j]) ** 2
    return error
# Computing new probability and its chi-error
def Calerr(rho,N,M,P,Pu):
    num_N=len(N)
    num_M=len(M)
    Pnew = np.zeros((num_N, num_M))
    for i in range(num_N):
        for j in range(num_M):
            Pnew[i][j] = np.real(np.trace(np.kron(N[i], M[j]) @ rho))
    return errP(Pnew,P,Pu)
def Cal_P(rho,N,M):
    num_N=len(N)
    num_M=len(M)
    Pnew = np.zeros((num_N, num_M))
    for i in range(num_N):
        for j in range(num_M):
            Pnew[i][j] = np.real(np.trace(np.kron(N[i], M[j]) @ rho))
    return Pnew
# See-saw optimization
def FitQ(P,Pu,rho,N,M):
    rho = optimize_rho(N, M, P, Pu,rho)   # Step 1: Optimize rho given N and M
    print(Calerr(rho, N, M,P,Pu))
    return rho