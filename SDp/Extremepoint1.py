import Bases
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from itertools import product
from pypoman import compute_polytope_vertices
import cdd
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


def construct_Mx(B,D):
    num_i = B.shape[1]
    num_j = D.shape[1]
    Mx = []
    for i in range(num_i):
        row = np.zeros(num_i * num_j)
        for j in range(num_j):
            row[i * num_j + j] = 1
        Mx.append(row)
    for k in range(B.shape[0]):
        for j in range(num_j):
            row = np.zeros(num_i * num_j)
            for i in range(num_i):
                row[i * num_j + j] = B[k, i]
            Mx.append(row)
    Mx = np.array(Mx)
    b_eq = np.concatenate((np.ones(num_i), np.zeros(B.shape[0] * num_j)))
    return Mx, b_eq

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


m, n = 20, 20
A=Bases.icosahedron_povm()
#A=generate_measurements(4,2)
D = Dvertices(A).T
# Compute M
bb= compute_M_trace(rho,A)
b = compute_M_trace(rho,A).flatten()
A_eq,b_eq = construct_Mx(B,D)

# Non-negativity constraints
A_ineq = np.eye(m * n)
b_ineq = np.zeros(m * n)

# Combine equality and inequality constraints into a single matrix
constraints = np.vstack([A_eq, A_ineq])
rhs = np.hstack([b_eq, b_ineq])

# Convert to cdd matrix, using float as the number type
mat = cdd.Matrix(np.hstack([rhs.reshape(-1, 1), constraints]), number_type='float')
mat.rep_type = cdd.RepType.INEQUALITY

# Run vertex enumeration
poly = cdd.Polyhedron(mat)
vertices = poly.get_generators()

# Extract vertices as numpy array
vertex_list = np.array(vertices)

# Display vertices
print("Vertices of the polytope are:")
print(vertex_list)
