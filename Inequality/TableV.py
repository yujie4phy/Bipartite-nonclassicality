import numpy as np
import gurobipy as gp
from gurobipy import Model, GRB
from scipy.spatial import HalfspaceIntersection
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import os
import Bases  # Your bases file


# =============================================================================
# PART 0: UTILS
# =============================================================================

def generate_random_mixed_state(dim_A, dim_B=None):
    if dim_B is None: dim_B = dim_A
    dim_total = dim_A * dim_B
    G = np.random.randn(dim_total, dim_total) + 1j * np.random.randn(dim_total, dim_total)
    rho = G @ G.conj().T
    return rho / np.trace(rho)


def is_entangled(rho, dim_A, dim_B=None):
    if dim_B is None: dim_B = dim_A
    rho_pt = rho.reshape(dim_A, dim_B, dim_A, dim_B).transpose(0, 3, 2, 1).reshape(dim_A * dim_B, dim_A * dim_B)
    return np.min(np.linalg.eigvalsh(rho_pt)) < -1e-12


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


# =============================================================================
# PART 1: From effects to noncontextual polytope, and compression
# =============================================================================


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


def compression(args):
    """
        1. Calculates Vertices (Dvertices)
        2. Performs SVD Compression (get_coords)
        3. Returns compressed coordinates
    """

    SVD_REL_TOL = 1e-13

    name, N, M = args
    Av, Bv = Dvertices(N), Dvertices(M)

    def get_coords(V):
        if len(V) == 0: return np.zeros((1, 1)), np.zeros((1, 1))
        U, svals, _ = np.linalg.svd(V.T, full_matrices=False)
        if svals.size == 0: return np.zeros((1, 1)), np.zeros((1, 1))

        tol = SVD_REL_TOL * svals[0]
        r = int(np.sum(svals > tol))
        r = max(r, 1)

        Q = U[:, :r]
        return Q, Q.T @ V.T

    QA, A_coords = get_coords(Av)
    QB, B_coords = get_coords(Bv)
    Q_AB = np.kron(QA, QB)

    return name, {"A_coords": A_coords, "B_coords": B_coords, "Q_AB": Q_AB, "N": N, "M": M}


# =============================================================================
# PART 2: STATE CHECKING
# =============================================================================

def check_states_block(args):
    """
       Given a set of state, checking violation againist a specific sets of operational constraints.
    """

    EPS_REJECT_NEG_PROB = 1e-9
    EPS_CLIP_NEG_PROB = 1e-12

    state_list, strategies = args
    local_counts = {name: 0 for name in strategies}
    processed_count = 0

    # Precompute per-strategy objects (depends only on strategy)
    M_small_by_strategy = {}
    AB_ops_by_strategy = {}
    u_small_by_strategy = {}

    for name, meta in strategies.items():
        A_c, B_c = meta["A_coords"], meta["B_coords"]
        dim_comp = A_c.shape[0] * B_c.shape[0]
        M_small = np.kron(A_c[:, :, None], B_c[:, None, :]).reshape(dim_comp, -1)
        M_small_by_strategy[name] = M_small

        N, M = meta["N"], meta["M"]
        AB_ops = [np.kron(A, B) for A in N for B in M]
        AB_ops_by_strategy[name] = AB_ops

        Q_AB = meta["Q_AB"]
        n_full = len(AB_ops)
        u_small_by_strategy[name] = Q_AB.T @ (np.ones(n_full) / n_full)

    # One Gurobi environment per process
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.setParam("Threads", 1)
        env.start()
    except:
        env = None

    # Build one LP per strategy (constraints fixed; objective changes per state)
    lp_by_strategy = {}
    for name, meta in strategies.items():
        M_small = M_small_by_strategy[name]
        u_small = u_small_by_strategy[name]
        try:
            model = gp.Model(env=env)
            model.Params.OutputFlag = 0
            model.Params.Method = 1
            model.Params.NumericFocus = 2
            model.Params.FeasibilityTol = 1e-9
            model.Params.OptimalityTol = 1e-9

            y = model.addMVar(shape=M_small.shape[0], lb=-GRB.INFINITY)
            model.addConstr(M_small.T @ y >= 0)
            model.addConstr(y @ u_small == 1)

            lp_by_strategy[name] = (model, y)
        except:
            lp_by_strategy[name] = None

    for rho in state_list:
        processed_count += 1

        for name, meta in strategies.items():
            pair = lp_by_strategy.get(name, None)
            if pair is None:
                continue

            model, y = pair
            Q_AB = meta["Q_AB"]
            AB_ops = AB_ops_by_strategy[name]

            p_full = np.array([np.real(np.trace(AB @ rho)) for AB in AB_ops], dtype=float)

            mn = float(np.min(p_full))
            if mn < -EPS_REJECT_NEG_PROB:
                continue
            if mn < 0:
                p_full[p_full < 0] = 0.0  # clip tiny negatives

            ssum = float(np.sum(p_full))
            if ssum <= 0:
                continue
            p_full /= ssum

            p_small = Q_AB.T @ p_full

            try:
                model.setObjective(y @ p_small, GRB.MINIMIZE)
                model.optimize()

                if model.status == GRB.OPTIMAL and model.ObjVal < -1e-6:
                    local_counts[name] += 1
            except:
                pass

    return local_counts, processed_count


# =============================================================================
# MAIN BLOCK
# =============================================================================

if __name__ == "__main__":
    DIM = 2
    NUM_STATES = 100000
    N_CORES = os.cpu_count() or 4


    BATCH_SIZE = 50

    # 1. Generate States
    print(f"Generating {NUM_STATES} states...", end=" ", flush=True)
    states = []
    while len(states) < NUM_STATES:
        rho = generate_random_mixed_state(DIM)
        if is_entangled(rho, DIM): states.append(rho)
    print("Done.")

    # 2. Define Strategies
    print("Defining strategies...")
    strats_config = []
    try:
        strats_config.append(("Square", Bases.Planor(4), Bases.Planortilde(4)))
        strats_config.append(("Cube", Bases.cube_povm(), Bases.cube_povm()))
        for nu in [1, 2]:
            strats_config.append((f"Geodesic v={nu}", Bases.geodesic_povm(nu), Bases.goldberg_povm(nu)))
    except AttributeError:
        pass

    # 3. PARALLEL GEOMETRY
    print("Compiling geometry (Parallel)...", end=" ", flush=True)
    strategies = {}
    with ProcessPoolExecutor() as executor:
        for name, data in executor.map(compression, strats_config):
            strategies[name] = data
    print("Done.\n")

    # 4. PARALLEL CHECKING (Dynamic Batches)
    batches = [states[i:i + BATCH_SIZE] for i in range(0, len(states), BATCH_SIZE)]
    tasks = [(batch, strategies) for batch in batches]

    print(f"Checking states with {len(batches)} batches...")
    start_check = time.time()

    total_counts = {name: 0 for name, _, _ in strats_config}
    total_processed = 0

    print(f"{'-' * 60}")
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_states_block, task) for task in tasks]
        for future in as_completed(futures):
            counts, n_processed = future.result()

            total_processed += n_processed
            for name, count in counts.items():
                total_counts[name] += count

            print(f"> Checked {total_processed}/{NUM_STATES} states...")

    elapsed = time.time() - start_check
    print(f"Total Computation Time: {elapsed:.2f}s\n")

    # 5. Results
    print(f"{'Strategy':<25} | {'Detected':<10} | {'Percentage':<10}")
    print("-" * 55)
    for name, count in total_counts.items():
        pct = (count / NUM_STATES) * 100
        print(f"{name:<25} | {count}/{NUM_STATES:<6} | {pct:.1f}%")
    print("=" * 60)
