import Bases
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from itertools import product
from pypoman import compute_polytope_vertices
from gurobipy import Model, GRB, LinExpr
import gurobipy as gp
def GellmannBasisElement(i, j, d):
    if i > j:  # symmetric elements
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = 1
        L[j - 1][i - 1] = 1
    elif i < j:  # antisymmetric elements
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = -1.0j
        L[j - 1][i - 1] = 1.0j
    elif i == j and i < d:  # diagonal elements
        L = np.sqrt(2 / (i * (i + 1))) * np.diag(
            [1 if n <= i else (-i if n == (i + 1) else 0) for n in range(1, d + 1)]
        )
    else:  # identity
        L = np.eye(d)
    return np.array(L / np.sqrt((L @ L).trace()))


def GelmannBasis(d):
    return [
        GellmannBasisElement(i, j, d) for i, j in product(range(1, d + 1), repeat=2)
    ]
def Dvertices(effects):
    d = effects[0].shape[0]
    basis = GelmannBasis(d)
    to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
    A=np.array([to_gellmann(v) for v in effects]).T.real
    alpha=null_space(A).T
    n = len(alpha[0])
    A_eq = np.vstack([np.ones(n), alpha])
    b_eq = np.concatenate([[1], np.zeros(len(alpha))])
    N = null_space(A_eq)
    A_eq_inv = np.linalg.pinv(A_eq)
    p0 = A_eq_inv @ b_eq
    A_ub = -np.eye(n)
    b_ub = np.zeros(n)
    A = A_ub @ N
    b = b_ub - A_ub @ p0
    vertices_lower_dim = compute_polytope_vertices(A, b)
    vertices = [p0 + N @ v for v in vertices_lower_dim]
    return np.array(vertices)
def compute_M_trace(A, B):
    n = len(A)  # Number of A matrices
    m = len(B)  # Number of B matrices
    M = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            product = np.dot(A[i], B[j])  # Matrix multiplication
            M[i, j] = np.abs(np.trace(product))  # Compute trace

    return M


def construct_Mx_and_tilde_b(B, D, b):
    num_i = B.shape[1]
    num_j = D.shape[1]
    Mx = []
    tilde_b = []
    for i in range(num_i):
        row = np.zeros(num_i * num_j)
        for j in range(num_j):
            row[i * num_j + j] = 1
        Mx.append(row)
        tilde_b.append(1)
    for k in range(B.shape[0]):
        for j in range(num_j):
            row = np.zeros(num_i * num_j)
            for i in range(num_i):
                row[i * num_j + j] = B[k, i]
            Mx.append(row)
            tilde_b.append(0)
    for i in range(num_i):
        for k in range(D.shape[0]):
            row = np.zeros(num_i * num_j)
            for j in range(num_j):
                row[i * num_j + j] = D[k, j]
            Mx.append(row)
            tilde_b.append(b[k])
    Mx = np.array(Mx)
    tilde_b = np.concatenate((np.ones(num_i), np.zeros(B.shape[0] * num_j), b))
    return Mx, tilde_b

def generate_measurements(k, dA):
    XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
    ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
    A = np.zeros((k, dA, dA), dtype=complex)
    for i in range(k):
        theta = (i) * 2*np.pi / k
        A[i, :, :] = 1/k * (np.eye(dA) + np.sin(theta) * XX + np.cos(theta) * ZZ)
    return A
def generate_states(k, dA):
    XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
    ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
    A = np.zeros((k, dA, dA), dtype=complex)
    for i in range(k):
        theta = (i) * 2*np.pi / k+np.pi/k
        A[i, :, :] = 1/k * (np.eye(dA) + np.sin(theta) * XX + np.cos(theta) * ZZ)
    return A



rho=Bases.dodecahedron_povm()
#rho=generate_states(4,2)
rho=rho*len(rho)/2
d = rho[0].shape[0]
basis = GelmannBasis(d)
to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
A = np.array([to_gellmann(v) for v in rho]).T.real
B = null_space(A).T

A=Bases.icosahedron_povm()
d = rho[0].shape[0]
basis = GelmannBasis(d)
to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
A = np.array([to_gellmann(v) for v in A]).T.real
C = null_space(A).T


# Define the size of the matrix
I = 12  # rows
J = 20  # columns

# Define decision variables
x = cp.Variable((I, J),nonneg=True)

# Define the B and D matrices
# You need to replace these with actual values of B and D

# Define the objective function
objective = -(x[7, 1] - x[7, 2] + x[10, 0] - x[10, 2] + x[11, 3] - x[11, 2])+sum(x[:,2])

# Define the constraints
constraints = []

# Constraint 1: sum_i x_ij = 1 for all j
for j in range(J):
    constraints.append(cp.sum(x[:, j]) == 1)

# Constraint 2: sum_i B_ki * x_ij = 0 for all k and j
for k in range(8):
    for j in range(J):
        constraints.append(cp.sum(C[k, :] * x[:, j]) == 0)

# Constraint 3: sum_j D_kj * x_ij = 0 for all k and i
for k in range(16):
    for i in range(I):
        constraints.append(cp.sum(B[k, :] * x[i, :]) == 0)

# Formulate the optimization problem
problem = cp.Problem(cp.Maximize(objective), constraints)

# Solve the problem
problem.solve()

# Output the results
print("Optimal value:", problem.value)
print("Optimal x matrix:")
print(x.value)