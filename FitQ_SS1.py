import cvxpy as cp
import numpy as np
import qutip as qt
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

def optimize_N(rho, M, P, Pu,N,pair1):
    num_N = len(N)
    num_M = len(M)
    dim_op = len(M[0])  # Dimension of N_i and M_j (2x2)
    """Optimize N given fixed rho and M, with specific pair constraints for N matrices."""
    # Define all num_N=12 complex Hermitian PSD variables for N
    N = [cp.Variable((dim_op, dim_op), hermitian=True) for _ in range(num_N)]
    objective = 0
    for i in range(num_N):
        for j in range(num_M):
            I_otimes_Mj = qt.tensor(qt.qeye(2), qt.Qobj(M[j]))  # Compute the tensor product and partial trace using qutip
            rho_full = I_otimes_Mj * qt.Qobj(rho, dims=[[2, 2], [2, 2]])
            rhoA = rho_full.ptrace([0]).full()  # Partial trace over system B, leaving only system A
            rhoA_cvx = cp.Constant(rhoA)
            objective += ((P[i, j] - cp.real(cp.trace(N[i] @ rhoA_cvx))) / Pu[i, j]) ** 2
    objective = cp.Minimize(objective)
    # Define the pairwise constraints for N matrices to sum to the identity matrix
    constraints = []
    for idx_1, idx_2 in pair1:
        constraints.append(N[idx_1] + N[idx_2] == cp.Constant(np.eye(dim_op)))
    for i in range(num_N):
        constraints.append(N[i] >> 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True)
    if prob.status not in ["infeasible", "unbounded"]:
        return [N[i].value for i in range(num_N)]
    else:
        print("Problem is infeasible or unbounded.")
        return None

def optimize_M(rho, N, P, Pu,M,pair2):
    num_N = len(N)
    num_M = len(M)
    dim_op = len(M[0])  # Dimension of N_i and M_j (2x2)
    """Optimize M given fixed rho and N with specific pair constraints."""
    M = [cp.Variable((dim_op, dim_op), hermitian=True) for _ in range(num_M)]
    objective = 0
    for j in range(num_M):
        for i in range(num_N):
            # Compute the tensor product and partial trace using qutip
            Ni_otimes_I = qt.tensor(qt.Qobj(N[i]), qt.qeye(2))
            rho_full = Ni_otimes_I * qt.Qobj(rho, dims=[[2, 2], [2, 2]])
            rhoB = rho_full.ptrace([1]).full()  # Partial trace over system A, leaving only system B
            rhoB_cvx = cp.Constant(rhoB)
            objective += ((P[i, j] - cp.real(cp.trace(M[j] @ rhoB_cvx))) / Pu[i, j]) ** 2
    objective = cp.Minimize(objective)
    # Add the constraints for the pairwise sums and positivity (PSD)
    constraints = []
    for idx_1, idx_2 in pair2:
        constraints.append(M[idx_1] + M[idx_2] == cp.Constant(np.eye(dim_op)))
    for j in range(num_M):
        constraints.append(M[j] >> 0)
    prob = cp.Problem(objective, constraints)
    prob.solve(warm_start=True)
    if prob.status not in ["infeasible", "unbounded"]:
        return [M[j].value for j in range(num_M)]
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
def FitQ(P,Pu,rho,N,M,pair1,pair2, max_iters=20,tolerance=1e-2):
    for iteration in range(max_iters):
        Calerr_old=Calerr(rho,N,M,P,Pu)
        rho = optimize_rho(N, M, P, Pu,rho)   # Step 1: Optimize rho given N and M
        N = optimize_N(rho, M, P, Pu,N,pair1)     # Step 2: Optimize N given rho and M
        M = optimize_M(rho, N, P, Pu,M,pair2)     # Step 3: Optimize M given rho and N
        print(Calerr(rho, N, M,P,Pu))
        # Check convergence (based on a change threshold)
        if abs(Calerr(rho, N, M,P,Pu)-Calerr_old) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break
    Pnew=Cal_P(rho,N,M)
    return Pnew