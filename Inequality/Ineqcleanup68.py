import numpy as np
import re
import itertools
import pickle
import sympy as sp


# --- Symmetry Functions ---
def gen_sym_perms():
    """
    Generate symmetry permutation matrices for a (6×8) inequality,
    where the vector of length 48 is reshaped into a (6,8) matrix.

    Now, we associate:
      - **Rows** with the 6 vertices of an octahedron → permutation matrices of shape 6×6.
      - **Columns** with the 8 vertices of a cube → permutation matrices of shape 8×8.

    It then computes, for each such 3×3 matrix, the induced permutation on the vertices:
      - For the octahedron (rows): using the vertex list
            [[1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
      - For the cube (columns): using the vertex list
            [[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],[-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]
    Returns:
       row_perms: list of 6×6 numpy arrays (permutation matrices for octahedron vertices)
       col_perms: list of 8×8 numpy arrays (permutation matrices for cube vertices)
    """
    # --- Step 1: Generate all full symmetry (3×3) matrices ---
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(perm):
                M[i, j] = signs[i]
            # Accept if |det(M)| = 1.
            if abs(round(np.linalg.det(M))) == 1:
                mats.append(M)
    unique_mats = {tuple(M.flatten()): M for M in mats}.values()
    full_mats = list(unique_mats)

    # --- Step 2: Define the vertices ---
    # For rows (we want a (6,8) inequality, rows correspond to octahedron vertices):
    oct_vertices = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]
    # For columns (cube vertices):
    cube_vertices = [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ]

    # --- Step 3: Helper inner functions ---
    def get_perm(M, verts):
        """Given a 3×3 matrix M and a list of vertices, return the induced permutation."""
        perm = []
        for v in verts:
            tv = M @ np.array(v)
            # Find the vertex that is nearly equal to tv.
            for i, w in enumerate(verts):
                if np.allclose(tv, w):
                    perm.append(i)
                    break
        return perm

    def make_perm_matrix(perm):
        """Given a permutation list, return the corresponding permutation matrix."""
        n = len(perm)
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[j, i] = 1
        return P

    # --- Step 4: Generate permutation matrices ---
    # For rows: use octahedron vertices
    row_mats = []
    for M in full_mats:
        perm_rows = get_perm(M, oct_vertices)
        P_row = make_perm_matrix(perm_rows)
        row_mats.append(P_row)
    unique_row_perms = {tuple(P.flatten()): P for P in row_mats}.values()

    # For columns: use cube vertices
    col_mats = []
    for M in full_mats:
        perm_cols = get_perm(M, cube_vertices)
        P_col = make_perm_matrix(perm_cols)
        col_mats.append(P_col)
    unique_col_perms = {tuple(P.flatten()): P for P in col_mats}.values()

    return list(unique_row_perms), list(unique_col_perms)


# --- Inequality Processing Functions ---
def canon_form(mat, rhs, rps, cps):
    """
    Apply all symmetries to the (6,8) inequality (mat, rhs) and return the lexicographically smallest candidate.
    Candidate is a tuple: (flatten(mat_sym), rhs).
    """
    cands = []
    for R in rps:
        for C in cps:
            t = R @ mat @ C
            cands.append((tuple(t.flatten()), rhs))
    return min(cands)


def reduce_eq(mat, rhs, eqs, dec=8):
    """
    Reduce the inequality (mat, rhs) modulo the equalities.
    Each inequality is represented as a 49-dimensional vector (flatten(mat) concatenated with rhs).
    Projects out the subspace spanned by the equality vectors.
    """
    v = np.concatenate([mat.flatten(), [rhs]])
    E_list = []
    for eq_mat, eq_rhs in eqs:
        eq_vec = np.concatenate([eq_mat.flatten(), [eq_rhs]])
        E_list.append(eq_vec)
    if E_list:
        E = np.array(E_list).T
        pinv = np.linalg.pinv(E.T @ E)
        P = E @ pinv @ E.T
        v_red = v - P @ v
    else:
        v_red = v
    v_red = np.round(v_red, decimals=dec)
    return v_red[:-1].reshape(mat.shape), v_red[-1]


def parse_file(fname):
    """
    Read file and extract equalities and inequalities.

    Assumes a line "DIM = 48" and an "INEQUALITIES_SECTION".
    The first block (with "==") contains equalities and the second (with "<=") contains inequalities.
    Each line gives a coefficient vector (length 48) which is reshaped to (6,8).
    Returns two lists: eqs, ineqs; each element is a tuple (matrix, rhs).
    """
    with open(fname, "r") as f:
        lines = f.readlines()

    dim = None
    for line in lines:
        if line.startswith("DIM"):
            dim = int(line.split("=")[1].strip())
            break
    if dim is None:
        raise ValueError("DIM not found")

    start = None
    for i, line in enumerate(lines):
        if "INEQUALITIES_SECTION" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("INEQUALITIES_SECTION not found")

    eq_lines = []
    ineq_lines = []
    read_eq = True
    for line in lines[start:]:
        if line.strip() == "":
            read_eq = False
            continue
        if line.startswith("("):
            if "==" in line and read_eq:
                eq_lines.append(line.strip())
            elif "<=" in line:
                ineq_lines.append(line.strip())

    if not eq_lines:
        raise ValueError("No equalities found")
    if not ineq_lines:
        raise ValueError("No inequalities found")

    def parse_line(line, dim):
        op = "==" if "==" in line else "<="
        parts = line.split(op)
        lhs = parts[0]
        rhs = parts[1].strip()
        if ")" in lhs:
            lhs = lhs.split(")", 1)[1].strip()
        coeff = np.zeros(dim, dtype=float)
        pattern = r"([+-]?\s*\d*)x(\d+)"
        terms = re.findall(pattern, lhs)
        for s, num in terms:
            s = s.replace(" ", "")
            if s == "" or s == "+":
                c = 1
            elif s == "-":
                c = -1
            else:
                c = float(s)
            coeff[int(num) - 1] = c
        return coeff, float(rhs)

    eqs = []
    for line in eq_lines:
        coeff, r = parse_line(line, dim)
        mat = coeff.reshape(6, 8)
        eqs.append((mat, r))

    ineqs = []
    for line in ineq_lines:
        coeff, r = parse_line(line, dim)
        mat = coeff.reshape(6, 8)
        ineqs.append((mat, r))

    return eqs, ineqs


def group_ineqs(fname, dec=8):
    """
    Read the file, reduce each inequality modulo the equalities,
    then use the full symmetry (via canonical form) to group inequalities.

    Returns:
       total: number of distinct orbits.
       reps: a list of representative inequalities (matrix, rhs) from the original ineq list.
       sizes: a list with the orbit sizes.
    """
    eqs, ineqs = parse_file(fname)
    rps, cps = gen_sym_perms()  # rps: 6x6, cps: 8x8
    red_ineqs = []
    for mat, r in ineqs:
        rm, rr = reduce_eq(mat, r, eqs, dec)
        red_ineqs.append((rm, rr))
    groups = {}
    for idx, (rm, rr) in enumerate(red_ineqs):
        cf = canon_form(rm, rr, rps, cps)
        groups.setdefault(cf, []).append(idx)
    reps = [ineqs[group[0]] for group in groups.values()]
    sizes = [len(group) for group in groups.values()]
    total = len(groups)
    return total, reps, sizes


def ineq_expr(y, rhs=0, ret_latex=False):
    """
    Given a flattened list y (length 48), reshape it to a (6,8) matrix and construct
    a symbolic inequality expression using sympy.
    """
    mat = np.array(y).reshape(6, 8)
    # Build tensor using fixed index maps.
    M_map = [(x, a) for x in range(3) for a in range(2)]
    N_map = [(y, 0) for y in range(4)] + [(y, 1) for y in reversed(range(4))]
    T = np.zeros((3, 4, 2, 2))
    for i, (x, a) in enumerate(M_map):
        for j, (y, b) in enumerate(N_map):
            T[x, y, a, b] = mat[i, j]
    # Create symbolic probability variables p(ab|xy).
    p = {}
    for x in range(3):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    p[(a, b, x, y)] = sp.Symbol(f'p({a}{b}|{x}{y})', real=True)
    expr = 0
    for x in range(3):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    expr += round(T[x, y, a, b] / 1.2) * p[(a, b, x, y)]
    ineq = sp.Ge(expr, rhs)
    return sp.latex(ineq) if ret_latex else str(ineq)


# --- Main ---
if __name__ == "__main__":
    fname = "Inequalities.txt"  # Adjust path as needed.
    tot, reps, sizes = group_ineqs(fname, dec=8)
    print("Total distinct orbits:", tot)
    for i, (r, s) in enumerate(zip(reps, sizes), 1):
        print(f"Orbit {i} (size {s}):")
        # Here we print the symbolic expression for the negative inequality
        # (if you wish to reverse the sign, as in your original code).
        print(ineq_expr(-r[0], -r[1], ret_latex=False))
        print("-" * 40)
    # Save the representatives using pickle.
    with open("ineq_list.pkl", "wb") as f:
        pickle.dump(reps, f)
    with open("ineq_list.pkl", "rb") as f:
        loaded = pickle.load(f)
    print("Loaded representatives; number of orbits:", len(loaded))