import numpy as np
import os
import concurrent.futures
from FitGPT_SS2 import FitGPT
# Import your modules/functions
from secondaryP import SecP
from FitQ_SS1 import FitQ
# from FitGPT_SS2 import FitGPT  # Uncomment if you wish to use GPT fitting instead.
import Bases
from Tofrequency import process_data
import Readfile
# Define the measurement pairings, file names, and paths
pair1 = [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]
pair2 = [(0, 18), (1, 15), (2, 16), (3, 11), (4, 10), (5, 14),
         (6, 12), (7, 17), (8, 19), (9, 13)]
r_values = [
0.27, 0.42, 0.43, 0.44,
0.45, 0.48, 0.50, 0.52,
0.54, 0.65, 0.75, 1
]
ROOT_DIR = "/Users/yujie4/Documents/Code/PycharmProjects/Bipartite-nonclasscality/Data/Counts_new"
# 2) Define the corresponding file names in the exact order:
file_names = [
    "counts_027.npy", "counts_042.npy", "counts_043.npy", "counts_044.npy",
    "counts_045.npy", "counts_048.npy", "counts_050.npy", "counts_052.npy",
    "counts_054.npy", "counts_065.npy", "counts_075.npy", "counts_100.npy"
]
def run_trial(data, rho, N, M, pair1, pair2, y_ineq):
    """
    Runs one trial of the experiment with Poisson noise added to the counts.
    """
    # Add Poisson noise to the original counts.
    # For each entry, treat the original count as the mean (lambda)
    noisy_data = np.random.poisson(lam=data)

    # Process the noisy counts into frequencies (P) and uncertainties (Pu)
    P, Pu = process_data(noisy_data)

    # Fit the data using the quantum fit routine.
    # (You can swap FitQ with FitGPT if desired.)
    Pfirst = FitQ(P, Pu, rho, N, M, pair1, pair2, max_iters=20, tolerance=1e-2)
    #Pfirst = FitGPT(P, Pu, rho, N, M, pair1, pair2, max_iters=100, tolerance=1e-2)  # fit to GPT
    # Obtain the second-order preparation
    Psecond = SecP(Pfirst, N, M)

    # Compute the quantum violation by taking the inner product.
    violation = np.inner(Psecond.flatten(), y_ineq)
    return violation


def monte_carlo_simulation(data, rho, N, M, pair1, pair2, y_ineq, num_trials=1):
    """
    Runs the experiment num_trials times (in parallel) and returns the list of violations.
    """
    violations = []
    # Use ProcessPoolExecutor to run trials in parallel.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit a trial for each simulation run.
        futures = [executor.submit(run_trial, data, rho, N, M, pair1, pair2, y_ineq)
                   for _ in range(num_trials)]

        # As each future completes, append its result.
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                violations.append(result)
            except Exception as exc:
                print(f"A trial generated an exception: {exc}")
    return violations
saveresults = []
for r_val, fname in zip(r_values, file_names):
    file_path = os.path.join(ROOT_DIR, fname)
    data = np.load(file_path)

    # Initial guesses for rho, and the measurement POVMs N and M.
    rho = np.array([[0.49 + 0.00j, 0, 0, 0.48],
                    [0, 0.01 + 0.00j, 0, 0],
                    [0, 0, 0.01 + 0.00j, 0],
                    [0.48, 0, 0, 0.49 + 0.00j]])
    M = Bases.dodecahedron_povm() * 10
    N = Bases.icosahedron_povm() * 6

    # Load and normalize the inequality array.
    y_ineq = np.load('y_ineq.npy')
    y_ineq = y_ineq / y_ineq[0]
    # Number of Monte Carlo trials
    if __name__ == '__main__':
        num_trials = 2
        # Run the Monte Carlo simulation.
        results = monte_carlo_simulation(data, rho, N, M, pair1, pair2, y_ineq, num_trials=num_trials)
        avg_violation = np.mean(results)
        std_violation=np.std(results, ddof=1)
        saveresults.append({
        'r_in': r_val,
        'filename': fname,
        'avg_violation': avg_violation,
        'std_violation': std_violation
         })
        # Compute the average violation over the trials.
        print("Individual violations from trials:", results)
        print("Average Quantum Violation:", avg_violation)
        print("Standard deviation:", std_violation)
for entry in saveresults:
    # print("ps= {:.3f}, fidelity= {:.5f}, po= {:.3f},  fidelity_opt = {:.5f}".format(
    #     entry['r_in'], entry['fidelity_o'],entry['optr'], entry['fidelity']))
    print("{:.3f},{:.6f}, {:.6}".format(
        entry['r_in'],entry['avg_violation'], entry['std_violation'],))
results_arr = np.array(saveresults, dtype=object)
np.save("violation-mento.npy", results_arr)
