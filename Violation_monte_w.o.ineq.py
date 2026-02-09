import numpy as np
import os
import concurrent.futures

import Bases
from Tofrequency import process_data
from secondaryP import SecP

from FitGPT_SS2 import FitGPT
from FitQ_SS1 import FitQ

from Farka1220P import compute_ineq_P, compute_eta_P
from Farka1220 import compute_ineq,compute_eta


# ---------------- settings ----------------

pair1 = [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]
pair2 = [(0, 18), (1, 15), (2, 16), (3, 11), (4, 10), (5, 14),
         (6, 12), (7, 17), (8, 19), (9, 13)]

r_values = [
    0.27, 0.42, 0.43, 0.44,
    0.45, 0.48, 0.50, 0.52,
    0.54, 0.65, 0.75, 1.00
]

ROOT_DIR = "/Users/yujie4/Documents/Code/PycharmProjects/Bipartite-nonclasscality/Data/Counts_new"

file_names = [
    "counts_027.npy", "counts_042.npy", "counts_043.npy", "counts_044.npy",
    "counts_045.npy", "counts_048.npy", "counts_050.npy", "counts_052.npy",
    "counts_054.npy", "counts_065.npy", "counts_075.npy", "counts_100.npy"
]

# fixed inequality for the "uniform-noise / fixed-ineq" eta
y_ineq = np.load("y_ineq.npy")
y_ineq = y_ineq / y_ineq[0]


# ---------------- one noisy trial ----------------

def run_trial_eta(data, rho0, N0, M0, pair1, pair2, y_ineq):
    """
    One Monte-Carlo trial:
      - add Poisson noise to counts
      - process to P, Pu
      - fit to get Pfirst
      - compute two etas:
          eta_gen  : from polytope-based method
          eta_fixed : from fixed inequality + uniform noise method
    """
    # assuming poisson noise for the photon statistics,  data is the raw data (counts)
    noisy_data = np.random.poisson(lam=data)
    P, Pu = process_data(noisy_data)

    Pfirst, N_opt, M_opt = Fitop(P, Pu, rho0, N0, M0, pair1, pair2,
                                 max_iters=20, tolerance=1e-2)
    # General noise robustness depends only on the distribution, without the need for the secondary statistics
    eta_gen = float(compute_eta_P(Pfirst))
    # Computing the secondary statistics for the noisy robustness to a specific inequality.
    Psecond = SecP(Pfirst, N0, M0)
    eta_fixed = float(compute_eta(Psecond , y_ineq))

    return 1-eta_gen, 1-eta_fixed


def monte_carlo_eta(data, rho0, N0, M0, pair1, pair2, y_ineq,
                    num_trials=2):
    """
    Run many noisy trials in parallel.
    Returns two lists: etas_gen (general robustness), etas_fixed (robustness to fixed inequality)
    """

    etas_gen, etas_fixed = [], []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_trial_eta, data, rho0, N0, M0, pair1, pair2, y_ineq)
            for _ in range(num_trials)
        ]
        for fut in concurrent.futures.as_completed(futures):
            try:
                ep, ef = fut.result()
                etas_gen.append(ep)
                etas_fixed.append(ef)
            except Exception as exc:
                print("Trial exception:", exc)

    return etas_gen, etas_fixed


# ---------------- main ---------------

if __name__ == "__main__":

    saveresults = []

    for r_val, fname in zip(r_values, file_names):

        file_path = os.path.join(ROOT_DIR, fname)
        data = np.load(file_path)
        ## initial conditions and targeted ideal measurements
        rho0 = np.array([[0.49 + 0.00j, 0, 0, 0.48],
                         [0, 0.01 + 0.00j, 0, 0],
                         [0, 0, 0.01 + 0.00j, 0],
                         [0.48, 0, 0, 0.49 + 0.00j]])
        M0 = Bases.dodecahedron_povm() * 10
        N0 = Bases.icosahedron_povm() * 6

        num_trials = 1
        etas_gen, etas_fixed = monte_carlo_eta(
            data, rho0, N0, M0, pair1, pair2, y_ineq,
            num_trials=num_trials
        )

        avg_gen= np.mean(etas_gen)
        std_gen = np.std(etas_gen, ddof=1) if len(etas_gen) > 1 else 0.0

        avg_fixed = np.mean(etas_fixed)
        std_fixed = np.std(etas_fixed, ddof=1) if len(etas_fixed) > 1 else 0.0

        saveresults.append({
            "r_in": r_val,
            "filename": fname,
            "avg_eta_gen": avg_gen,
            "std_eta_gen": std_gen,
            "avg_eta_fixed": avg_fixed,
            "std_eta_fixed": std_fixed
        })

        print(f"{fname}:")
        print("  etas_gen  =", etas_gen)
        print("  etas_fixed =", etas_fixed)
        print(f"  avg_gen   = {avg_gen:.6f}, std_gen   = {std_gen:.6f}")
        print(f"  avg_fixed  = {avg_fixed:.6f}, std_fixed  = {std_fixed:.6f}")
        print("-" * 60)

    # CSV-style summary
    for entry in saveresults:
        print("{:.3f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}".format(
            entry["r_in"],
            entry["avg_eta_gen"], entry["std_eta_gen"],
            entry["avg_eta_fixed"], entry["std_eta_fixed"]
        ))

    results_arr = np.array(saveresults, dtype=object)
    np.save("eta-monte-both.npy", results_arr)
