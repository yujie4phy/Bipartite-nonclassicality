import numpy as np
import cvxpy as cp
"""
This code implements a general positive map decomposition for an unsteerable state. It serves as a direct generalization of the methodology presented in the following repository:
https://github.com/mtcq/LHVextention/blob/main/FindLocalModel_BroadcastNL.m
"""
def partial_transpose(X, d, subsystem=1):
    """
    Computes the partial transpose of a matrix X acting on a bipartite system of dimension d x d.
    The subsystem parameter tells which subsystem to transpose (0 or 1).
    For subsystem=1, we map
      X[(i*d+j),(k*d+l)] --> X[(i*d+l),(k*d+j)]
    """
    D = d * d
    Y = cp.bmat([[None for j in range(d)] for i in range(d)])
    # We will fill the blocks where each block is d x d.
    # For each i,k for subsystem A and for subsystem B indices.
    Y_blocks = []
    for i in range(d):
        row_blocks = []
        for k in range(d):
            block = []
            for j in range(d):
                # Build each row of the block using transposition on subsystem index
                row = []
                for l in range(d):
                    # If transposing subsystem 1: swap the B indices: j and l.
                    if subsystem == 1:
                        # original index: row index = i*d + j, col index = k*d + l
                        # new index = i*d + l, col index = k*d + j
                        row.append(X[i * d + l, k * d + j])
                    else:
                        # Otherwise, for subsystem 0 (first subsystem), swap i and k:
                        row.append(X[k * d + j, i * d + l])
                # Convert row list to a cvxpy expression (concatenated horizontally)
                block.append(cp.hstack(row))
            row_blocks.append(cp.vstack(block))
        Y_blocks.append(row_blocks)
    # Now assemble the block matrix
    Y_full = cp.bmat(Y_blocks)
    return Y_full


def partial_trace_2(X, d):
    """
    Computes the partial trace over the second subsystem for an operator X defined on a bipartite system
    of dimension d x d. That is, Y = Tr_2[X] with Y_{ij} = sum_{k=0}^{d-1} X[i*d+k, j*d+k].
    """
    rows = []
    for i in range(d):
        row_expr = []
        for j in range(d):
            s = 0
            for k in range(d):
                s = s + X[i * d + k, j * d + k]
            row_expr.append(s)
        rows.append(cp.hstack(row_expr))
    return cp.vstack(rows)


def partial_trace_3(X, d, subsystem=1):
    """
    Computes the partial trace over one subsystem from an operator defined on a three-partite system
    with dimensions [d, d, d]. We assume that X is given as a matrix of shape (d**3, d**3)
    which corresponds to indices (i,k,l) for the rows and (j,m,n) for the columns.
    Here we trace over the second subsystem (subsystem index 1).
    The resulting operator is on a Hilbert space of dimension d x d.

    The mapping:
      i: subsystem 0, j: subsystem 0 (for columns)
      k: subsystem 1 (to be traced out)
      l: subsystem 2, n: subsystem 2 (for columns)

    Y[(i,d + l), (j,d + n)] = sum_{k=0}^{d-1} X[i*d*d + k*d + l, j*d*d + k*d + n]
    """
    # The output matrix will be of dimension (d*d, d*d)
    Y_entries = []
    for i in range(d):
        for l in range(d):
            row_list = []
            for j in range(d):
                for n in range(d):
                    s = 0
                    for k in range(d):
                        idx_row = i * d * d + k * d + l
                        idx_col = j * d * d + k * d + n
                        s = s + X[idx_row, idx_col]
                    row_list.append(s)
            Y_entries.append(cp.hstack(row_list))
    Y = cp.vstack(Y_entries)
    return Y


def CUdecomposition(rho_target):
    """
    Given a target quantum state rho_target (4x4 density matrix for two qubits),
    this function runs an SDP (via CVXPY) to determine whether rho_target can be written as a
    convex combination of a state with a local hidden variable (LHV) model and a PPT state.

    It prints the maximum value of eta found and whether the state has an LHV model.
    """
    # Dimension of each subsystem
    d = 2
    D = d * d  # dimension of two-qubit Hilbert space (4)

    # Best known isotropic state with LHV model.
    # Define the maximally entangled state |Phi+> (ketPhiP) and its density matrix PhiP.
    ketPhiP = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    PhiP = np.outer(ketPhiP, np.conjugate(ketPhiP))
    p_Designolle = 0.5
    IsotropicLocal = p_Designolle * PhiP + (1 - p_Designolle) * np.eye(D) / D

    # Compute the partial transpose of IsotropicLocal on the second subsystem.
    # In our helper this is implemented via the function partial_transpose.
    # Note: Because IsotropicLocal is a constant, we use its NumPy array in the kron operations below.
    IsoPT = np.empty_like(IsotropicLocal, dtype=complex)
    # We perform the partial transpose on IsotropicLocal using NumPy routines.
    for i in range(d):
        for j in range(d):
            for k in range(d):
                for l in range(d):
                    # Map: element at (i*d+j, k*d+l) becomes (i*d+l, k*d+j)
                    IsoPT[i * d + l, k * d + j] = IsotropicLocal[i * d + j, k * d + l]

    # Define the CVXPY variables.
    # We will force variables to be Hermitian by adding an equality constraint X == X.H.
    rho_PPT = cp.Variable((D, D), complex=True)
    map_CP1 = cp.Variable((D, D), complex=True)
    map_CP2 = cp.Variable((D, D), complex=True)
    eta = cp.Variable(nonneg=True)  # eta is a scalar (critical visibility)

    # Define constraints list.
    constraints = []

    # Enforce hermiticity and positive semidefiniteness.
    constraints += [rho_PPT == cp.conj(cp.transpose(rho_PPT))]
    constraints += [map_CP1 == cp.conj(cp.transpose(map_CP1))]
    constraints += [map_CP2 == cp.conj(cp.transpose(map_CP2))]
    constraints += [rho_PPT >> 0]
    constraints += [map_CP1 >> 0]
    constraints += [map_CP2 >> 0]
    # Compute the partial transpose of rho_PPT (using our helper function).
    rho_PPT_PT = partial_transpose(rho_PPT, d, subsystem=1)
    constraints += [rho_PPT_PT >> 0]

    # Define the map variable from CP maps.
    # map = map_CP1 + PartialTranspose(map_CP2)
    map_expr = map_CP1 + partial_transpose(map_CP2, d, subsystem=1)

    # Constraint: PartialTrace(map,2,[d,d]) == (trace(map)/d)*I
    PT_map = partial_trace_2(map_expr, d)
    # cp.trace(map_expr) returns a scalar
    constraints += [PT_map == (cp.trace(map_expr) / d) * np.eye(d)]

    kron_map = cp.kron(map_expr, np.eye(d))
    kron_I_isoPT = np.kron(np.eye(d), IsoPT)
    prod_expr = kron_map @ kron_I_isoPT
    # Now apply partial trace over the second subsystem for a tripartite system of dimensions [d,d,d].
    rho_local_expr = partial_trace_3(prod_expr, d, subsystem=1)

    # Final constraint: decompose the target state as convex combination.
    constraints += [eta * rho_target + (1 - eta) * np.eye(D) / D == rho_local_expr + rho_PPT]

    # (Optional) One may also constrain eta <= 1.
    constraints += [eta <= 1]

    # Set up and solve the SDP.
    prob = cp.Problem(cp.Maximize(eta), constraints)
    prob.solve(solver=cp.SCS)  # you can try other solvers if available

    # Print the optimum eta.
    print("Optimal eta =", eta.value)

    # Check the result.
    if eta.value is not None and eta.value >= 1 - 1e-6:
        print("The target state can be written as a convex combination of states with an LHV model.")
    else:
        print("No convex decomposition into an LHV model was found for the target state.")

