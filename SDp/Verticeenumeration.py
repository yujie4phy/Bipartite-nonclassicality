import numpy as np
from scipy.optimize import linprog
from itertools import product

# Define the number of conditions and outcomes
num_conditions = 6
num_outcomes = 6

# Define the alpha_y matrix manually or load it if provided
alpha_matrix = np.array([
    [1, -1, 1, -1, 1, -1],
    [1, -1, 0, 1, -1, 0],
    [1, 0, -1, 1, 0, -1]
])

# Define constraints for probability distributions
A_eq = np.zeros((num_conditions, num_conditions * num_outcomes))
for i in range(num_conditions):
    A_eq[i, i*num_outcomes:(i+1)*num_outcomes] = 1

b_eq = np.ones(num_conditions)

# Defining the non-negativity constraints
bounds = [(0, 1)] * (num_conditions * num_outcomes)

# Define the additional constraints
A_additional = np.kron(alpha_matrix, np.eye(num_outcomes))
b_additional = np.zeros(alpha_matrix.shape[0])

# Combine constraints
A_eq = np.vstack([A_eq, A_additional])
b_eq = np.concatenate([b_eq, b_additional])

# Use linear programming to find vertices (extreme points)
from scipy.spatial import ConvexHull

# Generate all combinations of outcomes where one outcome per condition is 1 and others are 0
combinations = np.eye(num_outcomes).tolist()
all_points = list(product(combinations, repeat=num_conditions))
all_points = np.array(all_points).reshape(-1, num_conditions * num_outcomes)

# Filter valid points that satisfy all constraints
valid_points = []
for point in all_points:
    if np.allclose(A_eq @ point, b_eq) and np.all(point >= 0):
        valid_points.append(point)

# Calculate vertices using Convex Hull
if valid_points:
    hull = ConvexHull(valid_points)
    vertices = np.array(valid_points)[hull.vertices]
    print("Vertices of the feasible region:")
    for vertex in vertices:
        print(vertex)
else:
    print("No valid points found.")
