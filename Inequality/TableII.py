import itertools
import os
import pickle
import re

import numpy as np
import sympy as sp


# -------------------------------------------------
#  Symmetry on rows (octahedron) and columns (cube)
# -------------------------------------------------
def gen_sym_perms():
    """
    Build permutation matrices for:
      - 6 vertices of an octahedron  → row permutations (6x6)
      - 8 vertices of a cube         → column permutations (8x8)

    We start from all signed permutation 3x3 matrices with |det| = 1
    and see how they permute the vertex sets.
    """

    # 1) All signed permutation matrices with |det| = 1
    mats = []
    for perm in itertools.permutations([0, 1, 2]):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3), dtype=int)
            for i, j in enumerate(perm):
                M[i, j] = signs[i]
            if abs(round(np.linalg.det(M))) == 1:
                mats.append(M)

    # Remove duplicates (there will be some)
    full_mats = list({tuple(M.flatten()): M for M in mats}.values())

    # Octahedron vertices (6 of them)
    oct_vertices = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
    ]

    # Cube vertices (8 of them)
    cube_vertices = [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1],
    ]

    def get_perm(M, verts):
        """Given a 3x3 matrix M, see how it permutes a list of vertices."""
        perm = []
        for v in verts:
            tv = M @ np.array(v)
            for i, w in enumerate(verts):
                if np.allclose(tv, w):
                    perm.append(i)
                    break
        return perm

    def make_perm_matrix(perm):
        """Turn a permutation list into a permutation matrix."""
        n = len(perm)
        P = np.zeros((n, n), dtype=int)
        for i, j in enumerate(perm):
            P[j, i] = 1
        return P

    # Row permutations: use octahedron vertices
    row_mats = []
    for M in full_mats:
        perm_rows = get_perm(M, oct_vertices)
        row_mats.append(make_perm_matrix(perm_rows))
    row_perms = list({tuple(P.flatten()): P for P in row_mats}.values())

    # Column permutations: use cube vertices
    col_mats = []
    for M in full_mats:
        perm_cols = get_perm(M, cube_vertices)
        col_mats.append(make_perm_matrix(perm_cols))
    col_perms = list({tuple(P.flatten()): P for P in col_mats}.values())

    return row_perms, col_perms


# -------------------------------------------------
#  Basic inequality manipulations
# -------------------------------------------------
def canon_form(mat, rhs, rps, cps):
    """
    Apply all row/column permutations and pick the lexicographically
    smallest representative. This labels the symmetry orbit.
    """
    candidates = []
    for R in rps:
        for C in cps:
            t = R @ mat @ C
            candidates.append((tuple(t.flatten()), rhs))
    return min(candidates)


def reduce_eq(mat, rhs, eqs, dec=8):
    """
    Reduce (mat, rhs) modulo the equalities.

    Represent each inequality as a length-49 vector (48 coeffs + rhs),
    project onto the orthogonal complement of the equality span,
    and round small numerical junk.
    """
    v = np.concatenate([mat.flatten(), [rhs]])

    if eqs:
        E_cols = []
        for eq_mat, eq_rhs in eqs:
            E_cols.append(np.concatenate([eq_mat.flatten(), [eq_rhs]]))
        E = np.array(E_cols).T  # shape (49, #eqs)

        pinv = np.linalg.pinv(E.T @ E)
        P = E @ pinv @ E.T       # projector onto span(E)
        v_red = v - P @ v        # component orthogonal to equalities
    else:
        v_red = v

    v_red = np.round(v_red, decimals=dec)
    return v_red[:-1].reshape(mat.shape), v_red[-1]


# -------------------------------------------------
#  Parsing the input file from Facets-style format
# -------------------------------------------------
def parse_file(fname):
    """
    Read `fname` and extract:
      - list of equalities  (mat, rhs)
      - list of inequalities (mat, rhs)

    Assumes:
      - a line 'DIM = 48'
      - a block starting with 'INEQUALITIES_SECTION'
      - equalities marked by '==' and inequalities by '<='
    """
    with open(fname, "r") as f:
        lines = f.readlines()

    # Get dimension
    dim = None
    for line in lines:
        if line.startswith("DIM"):
            dim = int(line.split("=")[1].strip())
            break
    if dim is None:
        raise ValueError("DIM not found in file")

    # Find start of inequalities section
    start = None
    for i, line in enumerate(lines):
        if "INEQUALITIES_SECTION" in line:
            start = i + 1
            break
    if start is None:
        raise ValueError("INEQUALITIES_SECTION not found")

    # Separate equality and inequality lines
    eq_lines = []
    ineq_lines = []
    read_eq = True  # we read equalities first, then inequalities

    for line in lines[start:]:
        s = line.strip()
        if not s:
            # blank line marks end of equality block
            read_eq = False
            continue
        if not s.startswith("("):
            continue

        if "==" in s and read_eq:
            eq_lines.append(s)
        elif "<=" in s:
            ineq_lines.append(s)

    if not eq_lines:
        raise ValueError("No equalities found")
    if not ineq_lines:
        raise ValueError("No inequalities found")

    def parse_line(line):
        """Parse a single line '(...): ... + ... x17 <= rhs' into coeff array and rhs."""
        op = "==" if "==" in line else "<="
        lhs, rhs = line.split(op)
        rhs = float(rhs.strip())

        # Drop the leading '(...) :' part
        if ")" in lhs:
            lhs = lhs.split(")", 1)[1].strip()

        coeff = np.zeros(dim, dtype=float)
        for s, num in re.findall(r"([+-]?\s*\d*)x(\d+)", lhs):
            s = s.replace(" ", "")
            if s in ("", "+"):
                c = 1.0
            elif s == "-":
                c = -1.0
            else:
                c = float(s)
            coeff[int(num) - 1] = c
        return coeff, rhs

    eqs = []
    for line in eq_lines:
        coeff, r = parse_line(line)
        eqs.append((coeff.reshape(6, 8), r))

    ineqs = []
    for line in ineq_lines:
        coeff, r = parse_line(line)
        ineqs.append((coeff.reshape(6, 8), r))

    return eqs, ineqs


# -------------------------------------------------
#  Group inequalities into symmetry orbits
# -------------------------------------------------
def group_ineqs(fname, dec=8):
    """
    - read eqs/ineqs from file
    - reduce each inequality modulo eqs
    - quotient by the full row/column symmetry

    Returns:
        total : number of orbits
        reps  : one representative (mat, rhs) per orbit (from the original list)
        sizes : orbit sizes
    """
    eqs, ineqs = parse_file(fname)
    row_perms, col_perms = gen_sym_perms()

    # First reduce modulo eqs
    reduced = []
    for mat, r in ineqs:
        rm, rr = reduce_eq(mat, r, eqs, dec)
        reduced.append((rm, rr))

    # Then group by symmetry
    groups = {}
    for idx, (rm, rr) in enumerate(reduced):
        key = canon_form(rm, rr, row_perms, col_perms)
        groups.setdefault(key, []).append(idx)

    reps = [ineqs[indices[0]] for indices in groups.values()]
    sizes = [len(indices) for indices in groups.values()]
    total = len(groups)

    return total, reps, sizes


# -------------------------------------------------
#  Turn a flattened inequality into a symbolic expression
# -------------------------------------------------
def ineq_expr(y, rhs=0, ret_latex=False):
    """
    Take y (length 48), reshape to 6x8, and write the inequality
    sum_x,y,a,b T_{x,y,a,b} p(ab|xy) >= rhs in sympy.

    The particular mapping of the 6x8 coefficients to (x,y,a,b)
    is the one you specified in your notes (M_map, N_map).
    """
    mat = np.array(y).reshape(6, 8)

    # Map rows/cols to (x,a) and (y,b)
    M_map = [(x, a) for x in range(3) for a in range(2)]
    N_map = [(y, 0) for y in range(4)] + [(y, 1) for y in reversed(range(4))]

    # T[x, y, a, b] holds the coefficient in front of p(ab|xy)
    T = np.zeros((3, 4, 2, 2))
    for i, (x, a) in enumerate(M_map):
        for j, (y, b) in enumerate(N_map):
            T[x, y, a, b] = mat[i, j]

    # Symbolic probabilities
    p = {}
    for x in range(3):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    p[(a, b, x, y)] = sp.Symbol(f"p({a}{b}|{x}{y})", real=True)

    # Build the linear expression; you can adjust the 1.2 factor if needed
    expr = 0
    for x in range(3):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    expr += round(T[x, y, a, b] / 1.2) * p[(a, b, x, y)]

    ineq = sp.Ge(expr, rhs)
    return sp.latex(ineq) if ret_latex else str(ineq)


# -------------------------------------------------
#  Main script: read file, group orbits, save reps
# -------------------------------------------------
if __name__ == "__main__":
    # Use the folder containing this script as the base for file IO.
    here = os.path.dirname(os.path.abspath(__file__))

    # Input file with the inequalities (adjust name if needed).
    ineq_file = os.path.join(here, "inequality68.txt")

    # Process all inequalities and group them into orbits.
    tot, reps, sizes = group_ineqs(ineq_file, dec=8)
    print("Total distinct orbits:", tot)

    # Print one representative inequality per orbit.
    for i, (r, s) in enumerate(zip(reps, sizes), start=1):
        print(f"Orbit {i} (size {s}):")
        # Note: I’m printing the *negative* inequality, as in your original code.
        print(ineq_expr(-r[0], -r[1], ret_latex=False))
        print("-" * 40)

    # Store the representatives to a pickle file for later use.
    pkl_path = os.path.join(here, "ineq_list.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(reps, f)

    # Quick sanity check: read them back.
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)
    print("Loaded representatives; number of orbits:", len(loaded))