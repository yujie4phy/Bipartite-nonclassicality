import numpy as np
from secondaryP import SecP
from FitQ_SS1 import FitQ
from FitGPT_SS2 import FitGPT
import Bases
data = np.load('p045.npy')  #  input raw data

pair1= [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]
pair2= [(0, 18), (1, 15), (2, 16), (3, 11), (4, 10), (5, 14), (6, 12), (7, 17), (8, 19), (9, 13)]
def swap1(b):
    row_swaps = [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]
    for row1, row2 in row_swaps:
        b[[row1, row2]] = b[[row2, row1]]
    return b
def swap2(b):
    column_swaps = [(0, 18), (1, 15), (2, 16), (3, 11), (4, 10), (5, 14), (6, 12), (7, 17), (8, 19), (9, 13)]
    for col1, col2 in column_swaps:
        b[:, [col1, col2]] = b[:, [col2, col1]]
    return b
t1 = data[0]; t2 = swap2(swap1(data[1])) ;t3 = swap1(data[2]); t4 = swap2(data[3])
P = (t1 + t2 + t3 + t4) / 4;    variance = np.var(np.stack([t1, t2, t3, t4]), axis=0)
Pu= np.sqrt(variance)+1e-6
# Initial guesses for rho, N, and M
rho = np.array([[0.47 + 0.00j, 0, 0, 0.48],
                [0, 0.01 + 0.00j, 0, 0],
                [0, 0, 0.01 + 0.00j, 0],
                [0.48, 0, 0, 0.51 + 0.00j]])
M = Bases.dodecahedron_povm() * 10
N = Bases.icosahedron_povm() * 6

# regularization of experiment data, convert frequency to quantum-(GPT-)compatible probability.
# P: frequency; Pu: uncertainty;  Intial guess for state:rho; Measurement N and M;
# pair1, pair2; paring the effect to measurements.

Pfirst=FitQ(P,Pu,rho,N,M,pair1,pair2, max_iters=20,tolerance=1e-2)  # fit to Quantum
#Pfirst=FitGPT(P,Pu,rho,N,M,pair1,pair2, max_iters=100,tolerance=1e-2) # fit to GPT

# given first-order preparation, return second-order preparation, satisfying specific operational equivelence.
Psecond=SecP(Pfirst,N,M)

# inequality for this specific scenario, could be adjust given other N and M
y_ineq = np.load('y_ineq.npy')

print("Quantum violation:", np.inner(Psecond.flatten(),y_ineq)) # negative value indicates quantum violation.

# A rough estimate of error
# (1) From monte carlo simulation;  (2)From multi trials of the same experiment. 
