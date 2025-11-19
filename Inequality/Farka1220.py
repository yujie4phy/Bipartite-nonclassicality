import Bases
import numpy as np
from itertools import product
from scipy.linalg import null_space
from pypoman import compute_polytope_vertices
from gurobipy import Model, GRB
import sympy as sp


# ---------------- Gell-Mann basis + vertices ----------------

def GellmannBasisElement(i, j, d):
    """Single (generalized) Gell-Mann matrix of size d×d."""
    if i > j:  # symmetric
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1, j - 1] = 1
        L[j - 1, i - 1] = 1
    elif i < j:  # antisymmetric
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1, j - 1] = -1.0j
        L[j - 1, i - 1] = 1.0j
    elif i == j and i < d:  # diagonal
        diag = [
            1 if n <= i else (-i if n == (i + 1) else 0)
            for n in range(1, d + 1)
        ]
        L = np.sqrt(2 / (i * (i + 1))) * np.diag(diag)
    else:  # identity
        L = np.eye(d)

    # Normalise w.r.t. Hilbert–Schmidt inner product
    return np.array(L / np.sqrt((L @ L).trace()))


def GellmannBasis(d):
    """Full d^2 Gell-Mann basis."""
    return [
        GellmannBasisElement(i, j, d)
        for i, j in product(range(1, d + 1), repeat=2)
    ]


def Dvertices(effects):
    """
    Given a POVM {E_k}, compute the vertices of the associated
    classical polytope in Gell-Mann coordinates.
    """
    d = effects[0].shape[0]
    basis = GellmannBasis(d)

    # Map each effect to Gell-Mann coordinates
    to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
    A = np.array([to_gellmann(v) for v in effects]).T.real

    # Linear relations between effects (null space)
    alpha = null_space(A).T
    n = len(alpha[0])

    # Constraints: p_k ≥ 0, sum_k p_k = 1, alpha · p = 0
    A_eq = np.vstack([np.ones(n), alpha])
    b_eq = np.concatenate([[1], np.zeros(len(alpha))])

    N = null_space(A_eq)
    A_eq_inv = np.linalg.pinv(A_eq)
    p0 = A_eq_inv @ b_eq  # particular solution

    # Vertex enumeration: A p ≤ b
    A_ub = -np.eye(n)
    b_ub = np.zeros(n)
    A_poly = A_ub @ N
    b_poly = b_ub - A_ub @ p0

    vertices_lower_dim = compute_polytope_vertices(A_poly, b_poly)
    vertices = [p0 + N @ v for v in vertices_lower_dim]
    return np.array(vertices)


# ---------------- Tensor vertices + M, b ----------------

def tensorvertices(A, B):
    """
    Given vertex sets A (n×d_a) and B (m×d_b), form all Kronecker products
    and view them as an (n m) × (d_a d_b) matrix.
    """
    n, dim_a = A.shape
    m, dim_b = B.shape
    C = np.zeros((n * m, dim_a * dim_b))

    idx = 0
    for i in range(n):
        for j in range(m):
            C[idx] = np.kron(A[i], B[j])
            idx += 1
    return C


def compute_M_trace(A_ops, B_ops):
    """
    Build M_{ij} = |Tr( A_i * B_j^T )| with a simple scaling.
    A_ops, B_ops are the POVM elements (e.g. icosahedron/dodecahedron effects).
    """
    n = len(A_ops)
    m = len(B_ops)
    M = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            # You can tweak the scaling factors if needed
            product = (A_ops[i] * len(A_ops)) @ (B_ops[j].T * len(B_ops)) / 8
            M[i, j] = abs(np.trace(product))

    return M


# ---------------- Dual LP ----------------

def solve_dual_problem(M, b):
    """
    Solve:  min_y   y·b
            s.t.    0 ≤ (yM)_j ≤ 1  for all j
    using Gurobi.
    """
    m, n = M.shape

    model = Model("Dual_Problem")
    y = model.addVars(m, lb=-GRB.INFINITY, name="y")

    # Objective: minimise sum_i b_i y_i
    model.setObjective(sum(b[i] * y[i] for i in range(m)), GRB.MINIMIZE)

    # Constraints: 0 ≤ (yM)_j ≤ 1
    for j in range(n):
        expr = sum(M[i, j] * y[i] for i in range(m))
        model.addConstr(expr >= 0, name=f"lower_{j}")
        model.addConstr(expr <= 1, name=f"upper_{j}")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return np.array([y[i].X for i in range(m)])
    else:
        print(f"Gurobi ended with status: {model.status}")
        return None


# ---------------- Inequality expression (240 coefficients) ----------------

def ineq_expr240_custom(data, rhs=0, ret_latex=False):
    """
    Map a length-240 vector into T[x,y,a,b] with
        x ∈ {0..5}, y ∈ {0..9}, a,b ∈ {0,1},
    then build  sum_{x,y,a,b} T[x,y,a,b] p(a b | x y) >= rhs.
    """
    # Row/column index maps (12 rows, 20 columns)
    M_map = [
        (0, 0), (0, 1),
        (1, 1), (1, 0),
        (0, 2), (0, 3),
        (1, 3), (1, 2),
        (0, 4), (0, 5),
        (1, 5), (1, 4),
    ]
    N_map = [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
        (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
        (1, 4), (1, 3), (1, 6), (1, 9), (1, 5),
        (1, 1), (1, 2), (1, 7), (1, 0), (1, 8),
    ]

    arr = np.array(data)
    if arr.size != 240:
        raise ValueError("Input data must have 240 elements.")
    M = arr.reshape(12, 20)

    # Fill tensor T[x, y, a, b]
    T = np.zeros((6, 10, 2, 2))
    for i in range(12):
        a,x = M_map[i]
        for j in range(20):
            b,y= N_map[j]
            T[x, y, a, b] = M[i, j]

    # Symbolic probabilities p(a b | x y)
    p = {}
    for x in range(6):
        for y in range(10):
            for a in range(2):
                for b in range(2):
                    p[(a, b, x, y)] = sp.Symbol(f"p({a}{b}|{x}{y})", real=True)

    # Build linear expression
    expr = 0
    for x in range(6):
        for y in range(10):
            for a in range(2):
                for b in range(2):
                    expr += T[x, y, a, b] * p[(a, b, x, y)]

    ineq = sp.Ge(expr, rhs)
    return sp.latex(ineq) if ret_latex else str(ineq)


# ---------------- Main script ----------------

def compute_ineq_from_povms(aa, bb, round_decimals=6):
    """
    Given POVMs aa (Alice) and bb (Bob), compute the normalised dual LP solution y_norm
    and print the corresponding symbolic Bell-type inequality.

    Returns:
        y_norm  -- the normalised and rounded inequality coefficients
    """

    # Convert POVM effects into classical-polytope vertices
    B = Dvertices(bb).T
    A = Dvertices(aa).T

    # Tensor-product vertices
    M = tensorvertices(A, B)

    # RHS vector from traces
    b = compute_M_trace(aa, bb).flatten()

    # Solve the dual LP
    y_solution = solve_dual_problem(M, b)
    if y_solution is None:
        print("Dual LP failed.")
        return None

    # Normalise coefficients so first entry is 1, remove small numerical noise
    y_norm = np.round(y_solution / y_solution[0], round_decimals)

    # Build and print the symbolic inequality
    ineq = ineq_expr240_custom(y_norm)
    sp.pprint(ineq, use_unicode=True)

    return y_norm


N = Bases.icosahedron_povm()
M = Bases.dodecahedron_povm()

# run computation
y_norm = compute_ineq_from_povms(N, M)

# now YOU decide how to save it
np.save("y_ineq1220.npy", y_norm)
