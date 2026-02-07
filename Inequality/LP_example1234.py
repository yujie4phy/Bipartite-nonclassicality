import numpy as np
from itertools import product
from scipy.spatial import HalfspaceIntersection
from gurobipy import Model, GRB


# import Bases  # Uncomment if you have your Bases.py file locally

# ---------------- 1. User's Gell-Mann Basis Logic ----------------

def GellmannBasisElement(i, j, d):
    """Single (generalized) Gell-Mann matrix of size dxd."""
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


# ---------------- 2. Generalized Geometric Vertex Enumeration ----------------
def Dvertices(effects, tol=1e-12):
    if len(effects) == 0:
        return np.array([])

    d = effects[0].shape[0]
    full_basis = GellmannBasis(d)

    basis_identity_idx = -1
    basis_traceless = []
    for k, B in enumerate(full_basis):
        if abs(np.trace(B)) > tol:
            basis_identity_idx = k
        else:
            basis_traceless.append(B)

    if basis_identity_idx == -1:
        raise ValueError("Basis does not contain an Identity-like element.")

    B_iso = full_basis[basis_identity_idx]
    c_iso = 1.0 / np.real(np.trace(B_iso))

    projection_map = []
    V = []
    consts = []

    for E in effects:
        v0 = np.real(np.trace(E @ B_iso))
        constant_term = v0 * c_iso
        v_vec = np.array([np.real(np.trace(E @ B)) for B in basis_traceless], dtype=float)

        projection_map.append((constant_term, v_vec))
        V.append(v_vec)
        consts.append(constant_term)

    V = np.array(V, dtype=float)            # (#effects, d^2-1)
    consts = np.array(consts, dtype=float)

    # Project to the span of {v_vec} (prevents unbounded / degenerate Qhull failures)
    U, s, _ = np.linalg.svd(V.T, full_matrices=False)
    rank = int(np.sum(s > 1e-10))
    if rank == 0:
        return np.array([[ct for (ct, _) in projection_map]], dtype=float)

    basis_sub = U[:, :rank]  # (d^2-1) x rank

    halfspaces = []
    proj_sub = []
    for (const, v_vec) in projection_map:
        v_sub = v_vec @ basis_sub
        proj_sub.append((const, v_sub))
        halfspaces.append(np.concatenate([-v_sub, [-const]]))  # A r + b <= 0

    halfspaces = np.array(halfspaces, dtype=float)

    # Strict interior point for Qhull
    r0 = np.zeros(rank, dtype=float)
    if np.min(consts) <= 1e-14:
        # maximize t s.t. const + v_sub·r >= t
        model = Model("FindInterior")
        model.Params.OutputFlag = 0
        model.Params.NumericFocus = 2
        r = model.addMVar(shape=rank, lb=-GRB.INFINITY)
        t = model.addVar(lb=-GRB.INFINITY)
        for (const, v_sub) in proj_sub:
            model.addConstr(const + v_sub @ r >= t)
        model.setObjective(t, GRB.MAXIMIZE)
        model.optimize()
        if model.status == GRB.OPTIMAL and t.X > 1e-12:
            r0 = r.X

    try:
        hs_solver = HalfspaceIntersection(halfspaces, r0)
        verts = hs_solver.intersections
    except Exception as e:
        print(f"Geometry Error (d={d}, rank={rank}): {e}")
        return np.array([])

    vertices_prob = []
    for r_sub in verts:
        p_vec = []
        for (const, v_sub) in proj_sub:
            val = const + float(np.dot(v_sub, r_sub))
            if val < -1e-9:
                raise ValueError(f"Negative probability {val} at a supposed vertex.")
            p_vec.append(0.0 if val < 0.0 else val)
        vertices_prob.append(p_vec)

    P = np.array(vertices_prob, dtype=float)
    P[np.abs(P) < 1e-14] = 0.0
    return P

# ---------------- 3. Generalized Quantum Physics ----------------

def compute_singlet_probabilities(A_ops, B_ops):
    """
    Computes P(a,b) = Tr((E_a x E_b) * rho).
    Uses Singlet for d=2, Maximally Entangled State for d>2.
    """
    d = A_ops[0].shape[0]

    if d == 2:
        psi = np.array([0, 1, -1, 0], dtype=np.complex128) / np.sqrt(2)
    else:
        psi = np.zeros(d * d, dtype=np.complex128)
        for i in range(d):
            psi[i * d + i] = 1.0
        psi /= np.sqrt(d)

    rho = np.outer(psi, psi.conj())
    probs = []

    for A in A_ops:
        for B in B_ops:
            E_joint = np.kron(A, B)
            p = np.real(np.trace(E_joint @ rho))
            probs.append(p)

    return np.array(probs)


# ---------------- 4p. Primal Gurobi Solver ----------------

def solve_primal_eta(M_small, p_target_small, p_noise_small):
    """
    Solves the Primal LP using an Augmented Matrix approach.
    This avoids the 'gurobipy._core.LinExpr' TypeError by vectorizing everything.

    We assume the equation:
       sum(lambda_i * v_i) = eta * p_target + (1 - eta) * p_noise

    Rearranging:
       sum(lambda_i * v_i) + eta * (p_noise - p_target) = p_noise

    We solve for vector x = [lambda_1, ..., lambda_n, eta]
    """
    dim_subspace, n_vertices = M_small.shape

    # 1. Prepare Flattened Vectors
    p_t = p_target_small.flatten()
    p_n = p_noise_small.flatten()
    diff_vec = p_n - p_t  # Shape (d,)

    # 2. Augment the Matrix: Add diff_vec as the last column
    # New shape: (d, n_vertices + 1)
    M_aug = np.hstack([M_small, diff_vec.reshape(-1, 1)])

    # 3. Setup Model
    model = Model("Primal_Eta_Membership")
    model.Params.OutputFlag = 0

    # Variable x includes [lambda_0 ... lambda_{n-1}, eta]
    # Length is n_vertices + 1
    x = model.addMVar(shape=n_vertices + 1, lb=0.0, ub=1.0, name="x")

    # 4. Objective: Maximize eta (the last element)
    model.setObjective(x[-1], GRB.MAXIMIZE)

    # 5. Constraint 1: Geometry
    # M_aug @ x == p_noise
    model.addConstr(M_aug @ x == p_n, name="Geometry")

    # 6. Constraint 2: Normalization (Sum of lambda == 1)
    # We sum x[:-1] (all elements except the last one)
    model.addConstr(x[:-1].sum() == 1, name="Normalization")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return x[-1].X
    else:
        return 0.0


# ---------------- 4d. DUAL Gurobi Solver  ----------------

def solve_dual_optimization(M_small, p_target_small, p_noise_small):
    """
    Solves the Dual LP to find the optimal Witness (Bell Inequality).

    Variables:
      y (vector): The coefficients of the witness in the compressed space.

    Constraints:
      1. y . V_local >= 0  (Witness must be positive on all local vertices)
      2. y . p_noise == 1  (Normalization to fix scale)

    Objective:
      Minimize y . p_target  (Find the most negative value possible)
    """
    dim_subspace, n_vertices = M_small.shape

    # 1. Setup Model
    model = Model("Dual_Eta_Witness")
    model.Params.OutputFlag = 0

    # y is unbounded (can be negative coefficients)
    y = model.addMVar(shape=dim_subspace, lb=-GRB.INFINITY, name="y")

    # 2. Objective: Minimize overlap with target state
    model.setObjective(y @ p_target_small, GRB.MINIMIZE)

    # 3. Constraint 1: Positivity on Local Polytope
    # M_small contains vertices as columns.
    # M_small.T is (n_vertices, dim).
    # (n_verts, dim) @ (dim, 1) -> (n_verts, 1) >= 0
    model.addConstr(M_small.T @ y >= 0, name="Positivity")

    # 4. Constraint 2: Normalization (White Noise = 1)
    model.addConstr(y @ p_noise_small == 1, name="Normalization")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        val_signal = model.ObjVal
        y_result = y.X

        # Calculate Eta from boundary condition
        # eta * val_signal + (1-eta) * 1 = 0  =>  eta = 1 / (1 - val_signal)

        if val_signal >= 1.0 - 1e-9:
            # If signal is >= noise, it's inside the polytope (or solver noise)
            return 1.0, None

        eta = 1.0 / (1.0 - val_signal)
        return eta, y_result
    else:
        return 0.0, None


# ---------------- 5. Helpers (Basis & Compression) ----------------

def local_basis_from_vertices(A_full, tol=1e-10):
    U, s, _ = np.linalg.svd(A_full.T, full_matrices=False)
    r = int(np.sum(s > tol))
    return U[:, :r], r


def build_joint_vertex_matrix_small(A_full, B_full, Q_A, Q_B):
    A_T = A_full.T
    B_T = B_full.T

    nA, VA = A_T.shape
    nB, VB = B_T.shape

    # Project vertices into low-d basis
    A_coords = Q_A.T @ A_T
    B_coords = Q_B.T @ B_T

    rA = A_coords.shape[0]
    rB = B_coords.shape[0]
    d = rA * rB

    M_small = np.zeros((d, VA * VB))

    idx = 0
    for i in range(VA):
        vec_a = A_coords[:, i]
        for j in range(VB):
            vec_b = B_coords[:, j]
            M_small[:, idx] = np.kron(vec_a, vec_b)
            idx += 1

    return M_small, rA, rB


# ---------------- 6p. Main (primal LP) ----------------

def compute_eta_primal(N, M):
    """
    Computes critical visibility eta directly using Primal LP.
    """
    # 1. Compute Local Vertices
    A_verts = Dvertices(N)
    B_verts = Dvertices(M)

    if len(A_verts) == 0 or len(B_verts) == 0:
        print("Error: Could not find vertices.")
        return 0.0

    print(f"Vertices found: Alice {A_verts.shape[0]}, Bob {B_verts.shape[0]}")

    # 2. Compute Bases
    Q_A, rA = local_basis_from_vertices(A_verts)
    Q_B, rB = local_basis_from_vertices(B_verts)
    Q_AB = np.kron(Q_A, Q_B)

    # 3. Build Compressed Matrix
    M_small, rA2, rB2 = build_joint_vertex_matrix_small(A_verts, B_verts, Q_A, Q_B)

    # 4. Compute Quantum Signal (Target)
    p_full = compute_singlet_probabilities(N, M)
    p_full = p_full / np.sum(p_full)  # Normalize

    # 5. Compute Noise (White Noise / Maximally Mixed)
    p_noise_full = np.ones_like(p_full) / len(p_full)

    # 6. Compress Signal and Noise
    p_target_small = Q_AB.T @ p_full
    p_noise_small = Q_AB.T @ p_noise_full

    # 7. Solve Primal LP
    eta = solve_primal_eta(M_small, p_target_small, p_noise_small)

    return eta


# ---------------- 6d. Main (Dual LP) ----------------

def compute_eta_dual(N, M):
    """
    Computes critical visibility eta AND the Witness S using Dual LP.
    """
    # 1. Compute Local Vertices
    A_verts = Dvertices(N)
    B_verts = Dvertices(M)

    if len(A_verts) == 0 or len(B_verts) == 0:
        print("Error: Could not find vertices.")
        return 0.0, None

    print(f"Vertices found: Alice {A_verts.shape[0]}, Bob {B_verts.shape[0]}")

    # 2. Compute Bases
    Q_A, rA = local_basis_from_vertices(A_verts)
    Q_B, rB = local_basis_from_vertices(B_verts)
    Q_AB = np.kron(Q_A, Q_B)

    # 3. Build Compressed Matrix
    M_small, rA2, rB2 = build_joint_vertex_matrix_small(A_verts, B_verts, Q_A, Q_B)

    # 4. Compute Quantum Signal (Target)
    p_full = compute_singlet_probabilities(N, M)
    p_full = p_full / np.sum(p_full)  # Normalize

    # 5. Compute Noise (White Noise / Maximally Mixed)
    p_noise_full = np.ones_like(p_full) / len(p_full)

    # 6. Compress Signal and Noise
    p_target_small = Q_AB.T @ p_full
    p_noise_small = Q_AB.T @ p_noise_full

    # 7. Solve DUAL LP
    eta, y_small = solve_dual_optimization(M_small, p_target_small, p_noise_small)

    # 8. Reconstruct Full Witness (Optional but useful)
    S_witness = None
    if y_small is not None:
        # Lift compressed witness y back to full space
        S_witness = Q_AB @ y_small

    return eta, S_witness


if __name__ == "__main__":
    # Example usage:
    try:
        import Bases

        # In our code, all measurements on Alice (Bob) are flag-convexfied to be a single POVM: {{M_{a|x}}_a}_x-->,{1/|X| M_{a|x}}_ax therefore, there is no explicit
        # Setting variable, concequently, we are just certifying classicality of the distribution p(abxy)=1/(|X||Y|) p(ab|xy), which is equivalent to certifing p(ab|xy).

        # Example 1
        #N, M = Bases.Planor(4), Bases.Planortilde(4)
        # Example 2
        # N,M=Bases.cube_povm(), Bases.octahedron_povm()

        # Example 3,4
        N, M = Bases.geodesic_povm(5), Bases.goldberg_povm(5)
        print("Computing Eta...")
        eta = compute_eta_primal(N, M)
        #eta, S = compute_eta_dual(N, M)
        print(f"Critical Visibility (eta): {eta}")
    except ImportError:
        print("Bases module not found. Please provide valid POVM lists N and M.")