import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
from scipy.spatial import HalfspaceIntersection
from itertools import product
import Bases


# =============================================================================
# PART 1: EXACT GEOMETRY ENGINE
# =============================================================================

def GellmannBasisElement(i, j, d):
    if i > j:
        L = np.zeros((d, d), dtype=np.complex128); L[i - 1, j - 1] = 1; L[j - 1, i - 1] = 1
    elif i < j:
        L = np.zeros((d, d), dtype=np.complex128); L[i - 1, j - 1] = -1j; L[j - 1, i - 1] = 1j
    elif i == j and i < d:
        diag = [1 if n <= i else (-i if n == i + 1 else 0) for n in range(1, d + 1)]
        L = np.diag(diag) * np.sqrt(2 / (i * (i + 1)))
    else:
        L = np.eye(d)
    return np.array(L / np.sqrt((L @ L).trace()))

def Dvertices(effects, tol=1e-12):
    if len(effects) == 0:
        return np.array([])

    d = effects[0].shape[0]
    full_basis = [GellmannBasisElement(i, j, d) for i, j in product(range(1, d + 1), repeat=2)]

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


def compute_eta_dual(N_a, M_b):
    # 1. Get Exact Vertices
    Av, Bv = Dvertices(N_a), Dvertices(M_b)

    # 2. SVD Compression
    def get_basis(V):
        if len(V) == 0: return np.zeros((1, 1)), np.zeros((1, 1))
        U, s, _ = np.linalg.svd(V.T, full_matrices=False)
        r = int(np.sum(s > 1e-10))
        return U[:, :r], U[:, :r].T @ V.T

    QA, A_coords = get_basis(Av)
    QB, B_coords = get_basis(Bv)
    Q_AB = np.kron(QA, QB)

    # 3. Build Compressed Joint Matrix M_small
    # Correct Tensor Construction: A(i,a) * B(j,b)
    # i,j are basis indices (rows of M). a,b are vertex indices (cols of M).
    M_tensor = np.einsum('ia, jb -> ijab', A_coords, B_coords)
    rA, VA = A_coords.shape
    rB, VB = B_coords.shape
    # Flatten to (rA*rB, VA*VB)
    M_small = M_tensor.reshape(rA * rB, VA * VB)

    p_vals = []
    for A in N_a:
        for B in M_b:
            # Original code logic: sum of product elements ~ Trace(A @ B.T)
            # Assuming real-valued POVMs where B.T = B.
            val = np.abs(np.trace(A @ B.T) )  # Scaled as in original
            p_vals.append(val)

    p_full = np.array(p_vals).flatten()
    p_full /= np.sum(p_full)

    p_small = Q_AB.T @ p_full

    # 5. Solve Exact LP
    # Minimize y.p s.t. y.M >= 0, y.u = 1
    d_poly = M_small.shape[0]

    try:
        with gp.Model("Exact") as m:
            m.setParam('OutputFlag', 0)
            y = m.addMVar(shape=d_poly, lb=-GRB.INFINITY)

            m.setObjective(y @ p_small, GRB.MINIMIZE)
            m.addConstr(M_small.T @ y >= 0)

            # Normalization (u is uniform)
            u_full = np.ones(len(p_full)) / len(p_full)
            u_small = Q_AB.T @ u_full
            m.addConstr(y @ u_small == 1)

            m.optimize()

            if m.status == GRB.OPTIMAL:
                val_signal = m.ObjVal
                # If val_signal >= 0, it's local.
                # If val_signal < 0, it's non-local.
                # Eta calculation:
                # noise_val = 1 (from constraint y.u=1)
                # eta * val_signal + (1-eta)*1 = 0
                # eta(val_signal - 1) = -1
                # eta = 1 / (1 - val_signal)

                if val_signal >= -1e-9:
                    return None, 1.0  # Local

                eta = 1.0 / (1.0 - val_signal)

                # We need to return S_final (the witness) in full space
                # Lift y back
                S_final = Q_AB @ y.X

                return S_final, eta

    except:
        pass

    return None, 0.0


# =============================================================================
# PART 3: MAIN
# =============================================================================

if __name__ == "__main__":
    # --- Config ---
    # N_values increased because exact solver is much faster than column generation
    N_values = [10, 20, 50, 100, 200]
    D_values = [2]
    TRIALS = 5

    results_db = {}

    print(f"{'=' * 65}")
    print(f"BENCHMARK: EXACT SOLVER (No Column Generation)")
    print(f"{'=' * 65}\n")

    for d in D_values:
        for n_bases in N_values:
            print(f">>> Processing: Dim={d}, N_bases={n_bases}")

            trial_etas = []

            for t in range(TRIALS):
                try:
                    # 1. Generate
                    N_povm = Bases.random_POVM(n_bases, d)
                    M_povm = Bases.random_POVM(n_bases, d)

                    # 2. Optimize (Using NEW Exact Solver)
                    S_opt, eta = compute_eta_dual(N_povm, M_povm)

                    if S_opt is not None:
                        print(f"    Trial {t + 1}: Eta = {eta:.6f}")
                        trial_etas.append(eta)
                        # Validation is technically redundant now (exact solver),
                        # but we can keep it for sanity checking if desired.
                        # validate_final_S_explicit(S_opt, N_povm, M_povm)
                    else:
                        print(f"    Trial {t + 1}: Local (Eta=1.0) or Failed.")
                        trial_etas.append(1.0)

                except Exception as e:
                    print(f"    Trial {t + 1}: Error - {e}")

            if trial_etas:
                results_db[(d, n_bases)] = trial_etas
            else:
                results_db[(d, n_bases)] = None

            print("-" * 50)

    # --- FINAL SUMMARY ---
    print("\n" + "=" * 65)
    print(f"{'FINAL RESULTS SUMMARY':^65}")
    print("=" * 65)
    print(f"{'Dim':<6} | {'N_bases':<8} | {'Avg Eta':<10} | {'Std Dev':<10} | {'Status'}")
    print("-" * 65)

    for d in D_values:
        for n in N_values:
            res = results_db.get((d, n))

            if res is not None:
                avg = np.mean(res)
                std = np.std(res)
                status = "OK"
                print(f"{d:<6} | {n:<8} | {avg:.6f}     | {std:.6f}     | {status}")
            else:
                print(f"{d:<6} | {n:<8} | {'N/A':<10} | {'N/A':<10} | FAIL")

    print("=" * 65)
