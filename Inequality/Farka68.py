import Bases
import cvxpy as cp
import numpy as np
from scipy.linalg import null_space
from itertools import product
from pypoman import compute_polytope_vertices
from gurobipy import Model, GRB, LinExpr
import sympy as sp
from decimal import Decimal, getcontext
import pickle
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

def computep(A, B):
    n = len(A)  # Number of A matrices
    m = len(B)  # Number of B matrices
    M = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            product = np.dot(A[i]*6, B[j].T*10)/2  # Matrix multiplication
            M[i, j] = np.abs(np.trace(product))  # Compute trace

    return M


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

def expression(y,rhs=0, return_latex=False):
    # Step 1: Reshape ymat into (6, 8)
    data_array = np.array(y).reshape(6, 8)

    # Step 2: Define index maps
    M_idx_map = [(x, a) for x in range(3) for a in range(2)]
    N_idx_map = [(y, 0) for y in range(4)] + [(y, 1) for y in reversed(range(4))]

    # Step 3: Fill the tensor
    tensor = np.zeros((3, 4, 2, 2))
    for i, (x, a) in enumerate(M_idx_map):
        for j, (y, b) in enumerate(N_idx_map):
            tensor[x, y, a, b] = data_array[i, j]

    # Step 4: Create symbolic variables p(ab|xy)
    p = {}
    for x in range(3):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    p[(a, b, x, y)] = sp.Symbol(f'p({a}{b}|{x}{y})', real=True)

    # Step 5: Build symbolic expression
    expr = 0
    for x in range(3):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    expr += round(tensor[x, y, a, b] / 1.2) * p[(a, b, x, y)]

    rhs=0
    inequality = sp.Ge(expr, rhs)  # expr >= rhs

    # Step 7: Return string or LaTeX form
    return sp.latex(inequality) if return_latex else str(inequality)
def save_conv_format(M: np.ndarray, filename: str = "vertice68.poi"):
    if M.shape != (48, 48):
        raise ValueError("Input array must be 48 by 48.")

    # Round each entry to the nearest integer
    M_rounded = np.rint(M).astype(int)

    with open(filename, 'w') as f:
        f.write("DIM=48\n\n")
        f.write("CONV_SECTION\n")

        for idx, row in enumerate(M_rounded, start=1):
            row_str = ' '.join(str(x) for x in row)
            f.write(f"({idx:3})  {row_str}\n")

        f.write("END\n")
# save_conv_format(M.T, "vertice68.poi")

bb=Bases.cube_povm()
B=Dvertices(bb).T

aa=Bases.octahedron_povm()
A= Dvertices(aa).T
# Compute M
M=tensorvertices(A, B)
b = computep(aa,bb).flatten()

y_solution = solve_dual_problem(M, b)
#ymat=y_solution.reshape(len(aa),len(bb))
expr=expression(y_solution)
# Display the symbolic expression
sp.pprint(expr, use_unicode=True)


np.save("y_ineq68.npy", y_solution)
