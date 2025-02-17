import numpy as np
import cvxpy as cp

def generate_measurements(N, k, dA):
    XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
    ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
    A = np.zeros((dA, dA, N, k), dtype=complex)

    for i in range(N):
        theta = (i) * np.pi / N
        A[:, :, i, 0] = 0.5 * (np.eye(dA) + np.sin(theta) * XX + np.cos(theta) * ZZ)
        A[:, :, i, 1] = np.eye(dA) - A[:, :, i, 0]

    return A

def jmdual(M_ax):
    dB = M_ax.shape[0]
    N = M_ax.shape[2]
    k = M_ax.shape[3]

    # Define variables for the coefficients of a steering inequality
    F_ax = [[cp.Variable((dB, dB), hermitian=True) for j in range(k)] for i in range(N)]

    tr_Fsig = 0
    trF_trsig = 0

    # Compute traces used in the objective function and constraints
    for i in range(N):
        for j in range(k):
            sig_ax = M_ax[:, :, i, j]
            tr_Fsig += cp.real(cp.trace(F_ax[i][j] @ sig_ax))
            trF_trsig += cp.real(cp.trace(F_ax[i][j]) * cp.trace(sig_ax))

    # Constraint setup
    constraints = []
    for l in range(k**N):
        U = cp.Constant(np.zeros((dB, dB)))
        string = np.base_repr(l, base=k, padding=N)[-N:]
        for i in range(N):
            c = int(string[i])
            for j in range(k):
                if c == j:
                    U += F_ax[i][j]

        constraints.append(0 >> U)

    constraints.append(1 - tr_Fsig + (1/dB) * trF_trsig == 0)

    # Define the objective function
    J = cp.Minimize(1 - cp.real(tr_Fsig))  # Ensure the objective is real

    # Create and solve the optimization problem
    problem = cp.Problem(J, constraints)
    problem.solve(solver=cp.MOSEK)

    # Convert F_ax from list of lists of CVXPY Variables to numpy array for output
    F_ax_value = np.array([[F_ax[i][j].value for j in range(k)] for i in range(N)])

    return F_ax_value, problem.value

# Example usage:
# Assuming M_ax is defined with appropriate dimensions and values
# F_ax, J = WNR_sdp1(M_ax)


# Example usage:
# Assuming M_ax is defined with appropriate dimensions and values
# F_ax, J = WNR_sdp1(M_ax)

N = 2
k = 2
dA = 2
A = generate_measurements(N, k, dA)
joint_M, eta_jm = jmdual(A)

print("Joint POVM (joint_M):")
print(joint_M)
print("Critical visibility (eta_jm):", eta_jm)

