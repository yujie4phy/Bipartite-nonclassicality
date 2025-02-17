import numpy as np
import cvxpy as cp

def generate_measurements(N, k_list, dA):
    XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
    ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
    A = np.zeros((dA, dA, N, max(k_list)), dtype=complex)

    for i in range(N):
        if k_list[i]==2:
            theta = (i) * np.pi / 3
            A[:, :, i, 0] = 0.5 * (np.eye(dA) + np.sin(theta) * XX + np.cos(theta) * ZZ)
            A[:, :, i, 1] = np.eye(dA) - A[:, :, i, 0]
        if k_list[i] == 3:
            theta = (i) * np.pi / 3
            A[:, :, i, 0] = 1/3 * (np.eye(dA) + np.sin(theta) * XX + np.cos(theta) * ZZ)
            A[:, :, i, 1] = 1/3 * (np.eye(dA) + np.sin(theta+2*np.pi / 3) * XX + np.cos(theta+2*np.pi / 3) * ZZ)
            A[:, :, i, 2] = np.eye(dA) - A[:, :, i, 0] - A[:, :, i, 1]
    return A

def jm(M, k_list):
    dA = M.shape[0]
    N = M.shape[2]

    # Calculate k^N where N is variable
    kN = np.prod(k_list)

    # Variables
    eta_jm = cp.Variable()
    joint_M = [cp.Variable((dA, dA), hermitian=True) for _ in range(kN)]

    # Constraints list
    constraints = [matrix >> 0 for matrix in joint_M]

    # Generate all deterministic probability distributions D for N inputs and variable outputs
    # Create a list of all possible outcomes combinations
    from itertools import product
    outcomes = list(product(*(range(k) for k in k_list)))

    D = np.zeros((N, max(k_list), kN))
    for l, outcome in enumerate(outcomes):
        for i, outcome_i in enumerate(outcome):
            D[i, outcome_i, l] = 1

    # Constraints for reproducing the depolarized measurement operators via coarse-graining of the joint POVM
    for i in range(N):
        for j in range(k_list[i]):
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
#A = generate_measurements(N, k_list, dA)
joint_M, eta_jm = jm(A, k_list)

print("Joint POVM (joint_M):")
print(joint_M)
print("Critical visibility (eta_jm):", eta_jm)
