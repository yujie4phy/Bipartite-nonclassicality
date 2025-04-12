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
            product = np.dot(A[i]*len(A), B[j].T*len(B))/8  # Matrix multiplication
            M[i, j] = np.abs(np.trace(product)) # Compute trace

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

def swap1(b):
    row_swaps = [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]
    for row1, row2 in row_swaps:
        b[[row1, row2]] = b[[row2, row1]]
    return b
def swap2(b):
    column_swaps = [(0, 18), (1, 15), (2, 16), (3, 11), (4, 10), (5, 14), (6, 12), (7, 17), (8, 19), (9, 13)]
    for col1, col2 in column_swaps:
        b[:, [col1, col2]] = b[:, [col2, col1]]
    return b


bb=Bases.dodecahedron_povm()
B=Dvertices(bb).T

aa=Bases.icosahedron_povm()
A= Dvertices(aa).T
# Compute M
M=tensorvertices(A, B)
data = np.load('probabilities.npy')
t1=data[0]
t2=swap2(swap1(data[1]))
t3=swap1(data[2])
t4=swap2(data[3])
b=(t1+t2+t3+t4)/4
b=b.flatten()
print(data)

b = compute_M_trace(aa,bb).flatten()
#b=data[1].flatten()
#M,b = construct_Mx_and_tilde_b(B, D, b)


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
        model.addConstr(sum(M[i, j] * y[i] for i in range(m)) <= 1, name=f"upper_constr_{j}")

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
ymat=y_solution.reshape(len(aa),len(bb))/y_solution[0]

import numpy as np
import sympy as sp


def ineq_expr240_custom(data, rhs=0, ret_latex=False):
    """
    Given a list (or 1D array) 'data' of 240 numbers, rearrange these numbers
    into a 4D tensor T of shape (6,10,2,2) using fixed index maps for the rows and columns,
    and build a symbolic inequality expression:

         ∑ₓ₌₀⁵ ∑ᵧ₌₀⁹ ∑ₐ₌₀¹ ∑ᵦ₌₀¹ T[x,y,a,b]*p(a,b|x,y) ≥ rhs.

    The fixed index maps are defined as follows:

       M_map (for rows):  -- 12 items (to be interpreted as a 6×2 grouping)
           [(0,0), (0,1),
            (1,1), (1,0),
            (0,2), (0,3),
            (1,3), (1,2),
            (0,4), (0,5),
            (1,5), (1,4)]

       N_map (for columns):  -- 20 items (to be interpreted as a 10×2 grouping)
           [(0,0), (0,1), (0,2), (0,3), (0,4),
            (0,5), (0,6), (0,7), (0,8), (0,9),
            (1,4), (1,3), (1,6), (1,9), (1,5),
            (1,1), (1,2), (1,7), (1,0), (1,8)]

    These maps imply that the 240 numbers are first reshaped into a 12×20 matrix.
    Then, for each row index i (0 ≤ i < 12) and column index j (0 ≤ j < 20):
         Let (x, a) = M_map[i]   and   (y, b) = N_map[j].
         Then assign T[x, y, a, b] = (data[i,j]).
    The resulting tensor T will have shape (6, 10, 2, 2), because:
         - 12 = 6 * 2  and  20 = 10 * 2.

    Finally, symbolic variables p(a,b|x,y) are generated (with x=0,...,5, y=0,...,9, a,b ∈ {0,1}),
    and the inequality expression is:
         sum_{x=0}^{5} sum_{y=0}^{9} sum_{a=0}^1 sum_{b=0}^1 T[x,y,a,b] * p(a,b|x,y) >= rhs.

    Parameters:
       data: list or 1D array of 240 numbers.
       rhs: right-hand side of the inequality (default 0).
       ret_latex: if True, return the LaTeX representation; otherwise return as string.

    Returns:
       The symbolic inequality expression as a string (or LaTeX if ret_latex is True).
    """
    # Fixed index maps:
    M_map = [(0, 0), (0, 1),
             (1, 1), (1, 0),
             (0, 2), (0, 3),
             (1, 3), (1, 2),
             (0, 4), (0, 5),
             (1, 5), (1, 4)]
    N_map = [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
             (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
             (1, 4), (1, 3), (1, 6), (1, 9), (1, 5),
             (1, 1), (1, 2), (1, 7), (1, 0), (1, 8)]

    # Step 1: Reshape data into a (12,20) matrix.
    arr = np.array(data)
    if arr.size != 240:
        raise ValueError("Input data must have 240 elements.")
    M = arr.reshape(12, 20)

    # Step 2: Build tensor T of shape (6,10,2,2).
    # Initialize T with zeros.
    T = np.zeros((6, 10, 2, 2))
    for i in range(12):
        (a, x) = M_map[i]  # each element of M_map is a tuple (x, a)
        for j in range(20):
            (b, y) = N_map[j]  # each element of N_map is a tuple (y, b)
            T[x, y, a, b] = M[i, j]

    # Step 3: Create symbolic variables p(a,b|x,y).
    p = {}
    for x in range(6):
        for y in range(10):
            for a in range(2):
                for b in range(2):
                    p[(a, b, x, y)] = sp.Symbol(f'p({a}{b}|{x}{y})', real=True)

    # Step 4: Build the symbolic expression.
    expr = 0
    for x in range(6):
        for y in range(10):
            for a in range(2):
                for b in range(2):
                    expr += T[x, y, a, b] * p[(a, b, x, y)]

    # Construct the inequality: expr >= rhs.
    inequality = sp.Ge(expr, rhs)

    return sp.latex(inequality) if ret_latex else str(inequality)


# --- Example usage ---
if __name__ == "__main__":
    # For demonstration, we'll create dummy data.
    # Replace this with your actual 240 data values.
    expr_str = ineq_expr240_custom(y_solution/y_solution[0], rhs=0, ret_latex=False)
    print(expr_str)