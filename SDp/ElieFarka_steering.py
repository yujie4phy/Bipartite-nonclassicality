import Bases
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from itertools import product
from pypoman import compute_polytope_vertices
from gurobipy import Model, GRB, LinExpr

from decimal import Decimal, getcontext

# Set precision for Decimal calculations
getcontext().prec = 10
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
def countzero(b_vectors, y, tolerance=1e-6):
    approx_zero_vectors = []
    for b in b_vectors:
        dot_product = np.dot(b, y)
        if np.abs(dot_product) <= tolerance:
            approx_zero_vectors.append(b)
    return np.array(approx_zero_vectors)
def checklp(vectors):
    if len(vectors) == 0:
        return 0, []
    # Stack vectors into a matrix
    matrix = np.vstack(vectors)
    # Calculate the rank of the matrix
    rank = np.linalg.matrix_rank(matrix)
    return rank

def compute_M_trace(A, B):
    n = len(A)  # Number of A matrices
    m = len(B)  # Number of B matrices
    M = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            product = np.dot(A[i]*6, B[j].T*10)/2  # Matrix multiplication
            M[i, j] = np.abs(np.trace(product))  # Compute trace

    return M



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

def tensorvertices(A, B):
    # Number of vertices in sets A and B
    n, dim_a = A.shape
    m, dim_b = B.shape
    # Initialize the new set of vertices C
    C = np.zeros((n * m, dim_a * dim_b))
    # Compute the tensor product for each pair of vertices from A and B
    index = 0
    for i in range(n):
        for j in range(m):
            C[index] = np.kron(A[i], B[j])
            index += 1
    return C

bb=Bases.dodecahedron_povm()
B=Dvertices(bb).T

aa=Bases.icosahedron_povm()
A= Dvertices(aa).T
# Compute M
M=tensorvertices(A, B)
b = compute_M_trace(aa,bb).flatten()


def solve_dual_problem(M, b):
    m, n = M.shape

    # Create a Gurobi model for the dual problem
    model = Model("Dual_Problem")

    # Add variables for y (no bound on y)
    y = model.addVars(m, lb=-GRB.INFINITY, name="y")

    # Set the objective function: Minimize y * b_star
    model.setObjective(sum(b[i] * y[i] for i in range(m)), GRB.MINIMIZE)

    # Add the constraints: 1 >= y * M >= 0
    for j in range(n):
        model.addConstr(sum(M[i, j] * y[i] for i in range(m)) >= 0, name=f"lower_constr_{j}")
        model.addConstr(sum(M[i, j] * y[i] for i in range(m)) <= 0.1, name=f"upper_constr_{j}")

    # Solve the model
    model.optimize()

    # Check if the solution is optimal
    if model.status == GRB.OPTIMAL:
        # Retrieve the solution for the dual variable y
        y_solution = np.array([y[i].X for i in range(m)])
        return y_solution
    else:
        print(f"Solver ended with status: {model.status}")
        return None

y_solution = solve_dual_problem(M, b)
ymat=y_solution.reshape(len(bb),len(aa))

zvectors = countzero(M.T, y_solution)

rank= checklp(zvectors)

print(rank)