import numpy as np
from scipy.linalg import null_space
from itertools import product
from scipy.optimize import linprog

def GellmannBasisElement(i, j, d):
    if i > j:  # symmetric elements
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = 1
        L[j - 1][i - 1] = 1
    elif i < j:  # antisymmetric elements
        L = np.zeros((d, d), dtype=np.complex128)
        L[i - 1][j - 1] = -1.0j
        L[j - 1][i - 1] = 1.0j
    elif i == j and i < d:  # diagonal elements
        L = np.sqrt(2 / (i * (i + 1))) * np.diag(
            [1 if n <= i else (-i if n == (i + 1) else 0) for n in range(1, d + 1)]
        )
    else:  # identity
        L = np.eye(d)
    return np.array(L / np.sqrt((L @ L).trace()))
def GelmannBasis(d):
    return [
        GellmannBasisElement(i, j, d) for i, j in product(range(1, d + 1), repeat=2)
    ]
def constrains(effects):
    effects=effects*len(effects)/2
    d = effects[0].shape[0]
    basis = GelmannBasis(d)
    to_gellmann = lambda v: np.array([(v @ e).trace() for e in basis[::-1]])
    A = np.array([to_gellmann(v) for v in effects]).T.real
    B = null_space(A)
    return B
def solveU(P, C):
    colb, rowu = P.shape
    _, ncons = C.shape
    # Objective: Maximize diagonal elements of U (equivalently, minimize negative sum of diagonal)
    c = [-1 if i == j else 0 for i in range(rowu) for j in range(rowu)]
    # Constraints
    A_eq = []
    b_eq = []
    # 1. Column sum constraints: sum_i u_ij = 1 for each j
    for j in range(rowu):
        colcons = [1 if k // rowu == j else 0 for k in range(rowu ** 2)]
        A_eq.append(colcons)
        b_eq.append(1)
    # 2. Zero constraint on double sum: sum_i sum_j P_ki * u_ij * C_jt = 0 for all k and t
    for k in range(colb):
        for t in range(ncons):
            constraint_row = [0] * (rowu ** 2)
            for i in range(rowu):
                for j in range(rowu):
                    index = i * rowu + j
                    constraint_row[index] = P[k, i] * C[j, t]
            A_eq.append(constraint_row)
            b_eq.append(0)
    A_eq = np.array(A_eq, dtype=float)
    b_eq = np.array(b_eq, dtype=float)
    # Bounds: 0 <= u_ij <= 1 (since we want a stochastic matrix)
    bounds = [(0, 1)] * (rowu ** 2)
    # Solve the linear program
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if res.success:
        u_optimal = res.x.reshape(rowu, rowu)
        largest_sum = -res.fun  # since we're maximizing, and c is negative
        return u_optimal, largest_sum
    else:
        print("No solution found:", res.message)
        return None, None
def SecP(P,MeasA,MeasB):
    consB = constrains(MeasB)
    consA = constrains(MeasA)
    u_op, rateB = solveU(P, consB)
    v_op, rateA = solveU(P.T, consA)
    print(rateB)
    print(rateA)
    if u_op is None:
        print("No second preparation on B")
    if v_op is None:
        print("No second preparation on A")
    if rateB < 0.8*len(consB):   #  Making sure the second preparation is not too bad
        print("Poor second preparation on B")
    if rateA < 0.8*len(consA):
        print("Poor second preparation on A")
    return v_op.T @ P @ u_op