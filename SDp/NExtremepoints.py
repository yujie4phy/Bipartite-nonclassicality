# Install pycddlib if you haven't already:
# pip install cddlib

import cdd
import numpy as np

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



effects=Bases.dodecahedron_povm()
#effects=generate_states(4,2)
effects=effects*len(effects)/2
d = effects[0].shape[0]
basis = GelmannBasis(d)
to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
A = np.array([to_gellmann(v) for v in effects]).T.real
B = null_space(A).T

A=Bases.icosahedron_povm()
#A=generate_measurements(4,2)
D = Dvertices(A).T
# Compute M
bb= compute_M_trace(effects,A)
b = compute_M_trace(effects,A).flatten()
M,b = construct_Mx_and_tilde_b(B, D, b)

A_ineq = np.eye(400)

b_ineq = np.zeros(400)

# Example equalities:
# x1 - x2 = 1

A_eq = M

b_eq = b

# Convert inequalities to cddlib format (b + A x >= 0)
num_ineq = A_ineq.shape[0]
h_ineq = np.hstack((b_ineq.reshape(-1, 1), -A_ineq))

# Convert equalities to cddlib format
num_eq = A_eq.shape[0]
h_eq = np.hstack((b_eq.reshape(-1, 1), -A_eq))

# Combine inequalities and equalities
h_rep = np.vstack((h_ineq, h_eq))

# Create a cddlib matrix
mat = cdd.Matrix(h_rep, number_type='float')

# Specify which rows are equalities
if num_eq > 0:
    mat.lin_set = set(range(num_ineq, num_ineq + num_eq))

poly = cdd.Polyhedron(mat)


generators = poly.get_generators()

vertices = []
for row in generators:
    if row[0] == 1:
        vertices.append(row[1:])

print("Extreme points (vertices):")
for v in vertices:
    print(np.array(v).astype(float))
