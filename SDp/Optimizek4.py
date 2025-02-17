import numpy as np
import gurobipy as gp
from scipy import integrate
from scipy.stats import ortho_group
from scipy.linalg import null_space
from scipy.optimize import linprog
from numpy.linalg import norm
import scipy
from copy import *


def find_largest_r(v1, v2, v3):
    def is_feasible(r,  vectors):
        A = np.array(vectors).T
        b1 = np.array([1, r, 0]) / 2
        b2 = np.array([1, 0, r]) / 2
        c = np.zeros(len(vectors))

        bounds = [(0, 1) for _ in range(len(vectors))]

        res1 = linprog(c, A_eq=A, b_eq=b1, bounds=bounds, method='highs')
        res2 = linprog(c, A_eq=A, b_eq=b2, bounds=bounds, method='highs')

        return res1.success and res2.success

    r = 0
    step = 0.01  # Small step to increase r

    while is_feasible(r, [v1, v2, v3]):
        r += step

    return r - step  # Return the last feasible r

rmax=0
# Example usage with v1, v2, and v3
# v1 = np.array([0.5, 0.3, -0.2])
# v2 = np.array([0.4, -0.3, 0.1])
# v3 = np.array([0.1, 0, 0.1])
def defineJA(JA1,JA2,JA3):
    me1=np.array([JA1,JA2,JA3])
    nsspace = scipy.linalg.null_space(np.array(me1).transpose()).transpose()
    if len(nsspace)>1:
        return [0,0]
    elif nsspace[0][0]>0 and nsspace[0][1]>0 and nsspace[0][2]>0:
        nu1=nsspace[0][0]/np.sum(nsspace[0])
        nu2=nsspace[0][1]/np.sum(nsspace[0])
        nu3=nsspace[0][2]/np.sum(nsspace[0])
    elif nsspace[0][0]<0 and nsspace[0][1]<0 and nsspace[0][2]<0:
        nu1=nsspace[0][0]/np.sum(nsspace[0])
        nu2=nsspace[0][1]/np.sum(nsspace[0])
        nu3=nsspace[0][2]/np.sum(nsspace[0])
    else:
        return [0,0]
    Ja1=nu1*np.array([1,0,0])+nu1*JA1
    Ja2=nu2*np.array([1,0,0])+nu2*JA2
    Ja3=nu3*np.array([1,0,0])+nu3*JA3
    JA=np.array([Ja1,Ja2,Ja3])
    return [1,JA]
for i in range(100000):
    a=2*np.pi*np.random.rand()
    da=2*np.pi/3
    b = 2*np.pi * np.random.rand()
    c = 2*np.pi * np.random.rand()
    JA1 = np.array([0,np.cos(a), np.sin(a)])
    JA2 = np.array([0,np.cos(b), np.sin(b)])
    JA3 = np.array([0,np.cos(c), np.sin(c)])
    bl,JA = defineJA(JA1, JA2, JA3)
    if bl == 1:
      r = find_largest_r(JA[0],JA[1], JA[2])
      if r>rmax:
          rmax=r
          print(rmax)
          print(JA)

print(f"Largest r: {rmax}")
