import numpy as np
from secondaryP import SecP
from Fitstatech import FitQ
from FitGPT_SS2 import FitGPT
import Bases
from Tofrequency import process_data
import os
import Isotrofit
import Readfile
from Perdurbation import perturbation
ROOT_DIR = "/Users/yujie4/Documents/Code/PycharmProjects/Bipartite-nonclasscality/Data/Counts_new"

# 1) Define the numeric parameter values (r) in a list:
r_values = [
0.27, 0.42, 0.43, 0.44,
0.45, 0.48, 0.50, 0.52,
0.54, 0.65, 0.75, 1
]

# 2) Define the corresponding file names in the exact order:
file_names = [
    "counts_027.npy", "counts_042.npy", "counts_043.npy", "counts_044.npy",
    "counts_045.npy", "counts_048.npy", "counts_050.npy", "counts_052.npy",
    "counts_054.npy", "counts_065.npy", "counts_075.npy", "counts_100.npy"
]
# Make sure r_values and file_names have the same length:
assert len(r_values) == len(file_names)

# 3) Loop over them in parallel:
results = []
# For N: 6 pairs (12 effects total)
N_pair = [[0, 3], [1, 2], [4, 7], [5, 6], [8, 11], [9, 10]]
# For M: 10 pairs (20 effects total)
M_pair = [[0, 18], [1, 15], [2, 16], [3, 11], [4, 10],
             [5, 14], [6, 12], [7, 17], [8, 19], [9, 13]]
m=np.load('m_angle.npy')
n=np.load('n_angle.npy')
for r_val, fname in zip(r_values, file_names):
    file_path = os.path.join(ROOT_DIR, fname)
    data = np.load(file_path)

    # Now you have the numeric handle r_val and the corresponding data
    P, Pu = process_data(data)

    # Example tomography or fit calls:
    # (Use your own initial guess for rho, or define it outside the loop)
    N = Bases.icosahedron_povm() * 6
    M = Bases.dodecahedron_povm() * 10
    optl = []
    fidelityl = []
    opt_rhol=[]
    fidelity_ol=[]
    for i in range(100):
        rho_init = r_val*np.array([
            [1/2, 0, 0, 1/2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1/2, 0, 0, 1/2]
        ])+1/4*(1-r_val)*np.identity(4)
        Mp = perturbation(m, M_pair, 1, 1)
        Np = perturbation(n, N_pair, 1, 1)
        rho = FitQ(P, Pu, rho_init, Np, Mp)
        optr, fidelity, fidelity_o, opt_rho = Isotrofit.Optimalr(rho, r_val)
        optl.append(optr)
        fidelityl.append(fidelity)
        opt_rhol.append(rho)
        fidelity_ol.append(fidelity_o)
        # print(f"Optimal r: {optr}")
        # print(f"Fidelity: {fidelity}")
    # Fit your rho:


    # Save or print your results
    results.append({
        'r_in': r_val,
        'filename': fname,
        'fidelity_o': np.mean(fidelity_ol),
        'fidelity_o-std': np.std(fidelity_ol),
        'rho_fitted': np.mean(opt_rhol,axis=0),
        'optr': np.mean(optl),
        'optr-std': np.std(optl),
        'opt_rho': opt_rho,
        'fidelity': np.mean(fidelityl),
        'fidelity-std': np.std(fidelityl),
    })

for entry in results:
    # print("ps= {:.3f}, fidelity= {:.5f}, po= {:.3f},  fidelity_opt = {:.5f}".format(
    #     entry['r_in'], entry['fidelity_o'],entry['optr'], entry['fidelity']))
    print("{:.3f}, {:.4f}, {:.4f}, {:.3f}, {:.3f},  {:.4f},  {:.4f}".format(
        entry['r_in'], entry['fidelity_o'],entry['fidelity_o-std'],entry['optr'], entry['optr-std'],entry['fidelity'],entry['fidelity-std'],))
results_arr = np.array(results, dtype=object)
np.save("results-mento.npy", results_arr)
