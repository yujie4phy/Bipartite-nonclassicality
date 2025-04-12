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
from Unsteeringdecomposition import Udecomposition
ROOT_DIR = "/Users/yujie4/Documents/Code/PycharmProjects/Bipartite-nonclasscality/Data/Counts"
results = []
# For N: 6 pairs (12 effects total)
N_pair = [[0, 3], [1, 2], [4, 7], [5, 6], [8, 11], [9, 10]]
# For M: 10 pairs (20 effects total)
M_pair = [[0, 18], [1, 15], [2, 16], [3, 11], [4, 10],
             [5, 14], [6, 12], [7, 17], [8, 19], [9, 13]]
m=np.load('m_angle.npy')
n=np.load('n_angle.npy')
fname= "counts_044.npy"
file_path = os.path.join(ROOT_DIR, fname)
data= np.load(file_path)
# Now you have the numeric handle r_val and the corresponding data
P, Pu = process_data(data)

# Example tomography or fit calls:
# (Use your own initial guess for rho, or define it outside the loop)
N = Bases.icosahedron_povm() * 6
M = Bases.dodecahedron_povm() * 10
for i in range(100):
    rho_init = 1/2*np.array([
        [1/2, 0, 0, 1/2],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [1/2, 0, 0, 1/2]
    ])+1/8*np.identity(4)
    Mp = perturbation(m, M_pair, 1, 1)
    Np = perturbation(n, N_pair, 1, 1)
    rho = FitQ(P, Pu, rho_init, Np, Mp)
    # decompose of tomography state into unsteerable state, using n_random=20 random unitary
    p_opt, sep_opt = Udecomposition(rho, 20)




