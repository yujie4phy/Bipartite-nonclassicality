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

def jm(M):
    dA = M.shape[0]
    N = M.shape[2]
    k = M.shape[3]
    kN = k ** N

    # Variables
    eta_jm = cp.Variable()
    # Define joint_M as a list of k^N dA x dA hermitian matrices
    joint_M = [cp.Variable((dA, dA), hermitian=True) for _ in range(kN)]

    # Constraints list
    constraints = []

    # Positivity constraints for the effects of the joint POVM
    for matrix in joint_M:
        constraints.append(matrix >> 0)

    # Generate all deterministic probability distributions D for N inputs and k possible outputs
    D = np.zeros((N, k, kN))
    for l in range(kN):
        string = np.base_repr(l, base=k, padding=N)[-N:]
        for i in range(N):
            c = int(string[i])
            D[i, c, l] = 1

    # Constraints for reproducing the depolarized measurement operators via coarse-graining of the joint POVM
    for i in range(N):
        for j in range(k):
            cg_ax = sum(D[i, j, l] * joint_M[l] for l in range(kN))
            constraints.append(cg_ax == eta_jm * M[:, :, i, j] + ((1 - eta_jm) / dA) * np.trace(M[:, :, i, j]) * np.eye(dA))

    # Objective function: maximize eta_jm
    objective = cp.Maximize(eta_jm)

    # Problem
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.MOSEK)

    # Retrieve the values of joint_M as numpy arrays for output
    joint_M_value = np.array([matrix.value for matrix in joint_M])

    return joint_M_value, eta_jm.value

# Example usage
N = 5
k = 2
dA = 2
A = generate_measurements(N, k, dA)
N = 5
k_list = [3, 3, 3,3,3]  # Example of varying number of outcomes
dA = 2
k=5
q=(1+np.sqrt(5))/2
XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
A5 = np.zeros((k, dA, dA), dtype=complex)
for i in range(k):
    theta = (i) * 2 * np.pi / k
    A5[i, :, :] =  (np.eye(dA) + np.sin(theta) * XX + np.cos(theta) * ZZ)
A = np.zeros((dA, dA, N, max(k_list)), dtype=complex)
A[:, :, 0, 0]=A5[0,:,:]/np.sqrt(5);A[:, :, 0, 1]=A5[2,:,:]/np.sqrt(5)/q;A[:, :, 0, 2]=A5[3,:,:]/np.sqrt(5)/q
A[:, :, 1, 0]=A5[1,:,:]/np.sqrt(5);A[:, :, 1, 1]=A5[3,:,:]/np.sqrt(5)/q;A[:, :, 1, 2]=A5[4,:,:]/np.sqrt(5)/q
A[:, :, 2, 0]=A5[2,:,:]/np.sqrt(5);A[:, :, 2, 1]=A5[4,:,:]/np.sqrt(5)/q;A[:, :, 2, 2]=A5[0,:,:]/np.sqrt(5)/q
A[:, :, 3, 0]=A5[3,:,:]/np.sqrt(5);A[:, :, 3, 1]=A5[0,:,:]/np.sqrt(5)/q;A[:, :, 3, 2]=A5[1,:,:]/np.sqrt(5)/q
A[:, :, 4, 0]=A5[4,:,:]/np.sqrt(5);A[:, :, 4, 1]=A5[1,:,:]/np.sqrt(5)/q;A[:, :, 4, 2]=A5[2,:,:]/np.sqrt(5)/q


joint_M, eta_jm = jm(A)

print("Joint POVM (joint_M):")
print(joint_M)
print("Critical visibility (eta_jm):", eta_jm)
