import cvxpy as cp
import numpy as np
import qutip as qt
import numpy as np
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*default will change to match NumPy's default order.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*Incorrect array format causing data to be copied.*",
    category=UserWarning,
)

# ------------------------
#   Subroutine: optimize_rho
# ------------------------
def optimize_rho(N, M, P, Pu, rho):
    """
    Optimize rho given fixed N and M, by minimizing chi^2 between
    P(i,j) and tr[(N_i \otimes M_j) rho].
    """
    num_N = len(N)
    num_M = len(M)
    dim_rho = len(rho)  # e.g. 4 for 2-qubit

    # CVXPY variable for rho
    rho_var = cp.Variable((dim_rho, dim_rho), hermitian=True)

    # Precompute (N_i \otimes M_j) for speed
    NiMj = []
    for i in range(num_N):
        for j in range(num_M):
            NiMj.append(np.kron(N[i], M[j]))  # 4x4 for two-qubit

    # Build the residuals in a list, then sum squares
    residuals = []
    idx = 0
    for i in range(num_N):
        for j in range(num_M):
            # Predicted probability = real(trace((N_i \otimes M_j) rho))
            pred_ij = cp.real(cp.trace(cp.Constant(NiMj[idx]) @ rho_var))
            res_ij = (P[i, j] - pred_ij) / Pu[i, j]
            residuals.append(res_ij)
            idx += 1

    objective = cp.sum_squares(cp.hstack(residuals))  # sum of squares of residuals

    # Constraints: rho >= 0, trace(rho)=1
    constraints = [
        rho_var >> 0,
        cp.trace(rho_var) == 1
    ]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    # You can choose your solver; MOSEK is shown if licensed
    prob.solve(warm_start=True, solver=cp.MOSEK)

    if prob.status not in ["infeasible", "unbounded"]:
        return rho_var.value
    else:
        print("optimize_rho: Problem infeasible or unbounded.")
        return None


# ------------------------
#   Subroutine: optimize_N
# ------------------------
def optimize_N(rho, M, P, Pu, N, pair1):
    """
    Optimize N given fixed rho, M.
    pair1: list of index pairs (idx1, idx2) that must sum to the identity.
    """
    num_N = len(N)
    num_M = len(M)
    dim_op = len(N[0])  # e.g. 2 for single-qubit ops

    # CVXPY variables for each N_i
    N_vars = [cp.Variable((dim_op, dim_op), hermitian=True) for _ in range(num_N)]

    # Precompute partial trace over B:  rho^A_j = Tr_B[(I \otimes M_j) rho]
    # This depends only on j
    partialA = []
    rho_qobj = qt.Qobj(rho, dims=[[2, 2], [2, 2]])
    for j in range(num_M):
        # I \otimes M_j
        I_otimes_Mj = qt.tensor(qt.qeye(2), qt.Qobj(M[j]))
        # Multiply by rho, then partial trace over Bob ([1])
        rho_full = I_otimes_Mj * rho_qobj
        rhoA_j = rho_full.ptrace([0]).full()  # 2x2 matrix
        partialA.append(cp.Constant(rhoA_j))

    # Build residuals
    residuals = []
    for i in range(num_N):
        for j in range(num_M):
            pred_ij = cp.real(cp.trace(N_vars[i] @ partialA[j]))
            res_ij = (P[i, j] - pred_ij) / Pu[i, j]
            residuals.append(res_ij)

    objective = cp.sum_squares(cp.hstack(residuals))

    # Constraints:
    #  1) positivity: N_i >= 0
    #  2) for each pair (idx1, idx2), N[idx1] + N[idx2] = I
    constraints = []
    for i in range(num_N):
        constraints.append(N_vars[i] >> 0)
    for (idx1, idx2) in pair1:
        constraints.append(N_vars[idx1] + N_vars[idx2] == cp.Constant(np.eye(dim_op)))

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(warm_start=True)

    if prob.status not in ["infeasible", "unbounded"]:
        # Return a plain list of optimized N_i
        return [N_vars[i].value for i in range(num_N)]
    else:
        print("optimize_N: Problem infeasible or unbounded.")
        return None


# ------------------------
#   Subroutine: optimize_M
# ------------------------
def optimize_M(rho, N, P, Pu, M, pair2):
    """
    Optimize M given fixed rho, N.
    pair2: list of index pairs (idx1, idx2) that must sum to identity.
    """
    num_N = len(N)
    num_M = len(M)
    dim_op = len(M[0])

    # CVXPY variables for each M_j
    M_vars = [cp.Variable((dim_op, dim_op), hermitian=True) for _ in range(num_M)]

    # Precompute partial trace over A: rho^B_i = Tr_A[(N_i \otimes I) rho]
    partialB = []
    rho_qobj = qt.Qobj(rho, dims=[[2, 2], [2, 2]])
    for i in range(num_N):
        Ni_otimes_I = qt.tensor(qt.Qobj(N[i]), qt.qeye(2))
        rho_full = Ni_otimes_I * rho_qobj
        rhoB_i = rho_full.ptrace([1]).full()  # 2x2
        partialB.append(cp.Constant(rhoB_i))

    # Build residuals
    residuals = []
    for j in range(num_M):
        for i in range(num_N):
            pred_ij = cp.real(cp.trace(M_vars[j] @ partialB[i]))
            res_ij = (P[i, j] - pred_ij) / Pu[i, j]
            residuals.append(res_ij)

    objective = cp.sum_squares(cp.hstack(residuals))

    # Constraints:
    #  1) positivity: M_j >= 0
    #  2) for each pair (idx1, idx2), M[idx1] + M[idx2] = I
    constraints = []
    for j in range(num_M):
        constraints.append(M_vars[j] >> 0)
    for (idx1, idx2) in pair2:
        constraints.append(M_vars[idx1] + M_vars[idx2] == cp.Constant(np.eye(dim_op)))

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(warm_start=True)

    if prob.status not in ["infeasible", "unbounded"]:
        return [M_vars[j].value for j in range(num_M)]
    else:
        print("optimize_M: Problem infeasible or unbounded.")
        return None


# ------------------------
#   Error Function: errP
# ------------------------
def errP(Pnew, P, Pu):
    """
    Compute sum of squared residuals with uncertainties:
      sum_{i,j} [ (P[i,j] - Pnew[i,j]) / Pu[i,j] ]^2
    """
    return np.sum(((P - Pnew) / Pu) ** 2)


# ------------------------
#   Compute chi^2 for current (rho,N,M)
# ------------------------
def Calerr(rho, N, M, P, Pu):
    """
    Re-compute the chi^2 = sum_{i,j} [ (P - trace((N_i \otimes M_j)rho)) / Pu ]^2
    """
    num_N = len(N)
    num_M = len(M)
    Pnew = np.zeros((num_N, num_M))
    for i in range(num_N):
        for j in range(num_M):
            Pnew[i, j] = np.real(np.trace(np.kron(N[i], M[j]) @ rho))
    return errP(Pnew, P, Pu)


# ------------------------
#   Compute predicted probability table
# ------------------------
def Cal_P(rho, N, M):
    """
    Return the table Pnew[i,j] = trace[(N_i \otimes M_j) rho].
    """
    num_N = len(N)
    num_M = len(M)
    Pnew = np.zeros((num_N, num_M))
    for i in range(num_N):
        for j in range(num_M):
            Pnew[i, j] = np.real(np.trace(np.kron(N[i], M[j]) @ rho))
    return Pnew


# ------------------------
#   Main See-Saw Loop
# ------------------------
def FitQ(P, Pu, rho, N, M, pair1, pair2, max_iters=20, tolerance=1e-2):
    """
    Iterative see-saw:
      1) Optimize rho (fix N, M)
      2) Optimize N   (fix rho, M)
      3) Optimize M   (fix rho, N)
      until convergence or max_iters
    """

    for iteration in range(max_iters):
        old_err = Calerr(rho, N, M, P, Pu)

        # Step 1
        rho_opt = optimize_rho(N, M, P, Pu, rho)
        if rho_opt is None:
            break
        rho = rho_opt

        # Step 2
        N_opt = optimize_N(rho, M, P, Pu, N, pair1)
        if N_opt is None:
            break
        N = N_opt

        # Step 3
        M_opt = optimize_M(rho, N, P, Pu, M, pair2)
        if M_opt is None:
            break
        M = M_opt

        new_err = Calerr(rho, N, M, P, Pu)
        print("Iteration:", iteration, "chi^2 =", new_err)

        # Convergence check
        if abs(new_err - old_err) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

    # Return final predicted probabilities
    return Cal_P(rho, N, M)


# fit to Quantum and return optimal setting.
def Fitop(P, Pu, rho, N, M, pair1, pair2, max_iters=20, tolerance=1e-2):
    """
    Iterative see-saw:
      1) Optimize rho (fix N, M)
      2) Optimize N   (fix rho, M)
      3) Optimize M   (fix rho, N)
      until convergence or max_iters

      return the operational identity for the optimized sets of measurements N and M.
    """

    for iteration in range(max_iters):
        old_err = Calerr(rho, N, M, P, Pu)

        # Step 1
        rho_opt = optimize_rho(N, M, P, Pu, rho)
        if rho_opt is None:
            break
        rho = rho_opt

        # Step 2
        N_opt = optimize_N(rho, M, P, Pu, N, pair1)
        if N_opt is None:
            break
        N = N_opt

        # Step 3
        M_opt = optimize_M(rho, N, P, Pu, M, pair2)
        if M_opt is None:
            break
        M = M_opt

        new_err = Calerr(rho, N, M, P, Pu)
        print("Iteration:", iteration, "chi^2 =", new_err)

        # Convergence check
        if abs(new_err - old_err) < tolerance:
            print(f"Converged after {iteration} iterations.")
            break

    # Return final predicted probabilities
    return Cal_P(rho, N, M), N,M
