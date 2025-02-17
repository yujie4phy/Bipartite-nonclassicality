import numpy as np

# Define the imaginary unit
i = 1j

# Basis vectors
M0 = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
])

M1 = np.array([
    [1, 1, 1, 1],
    [1, -1, 1, -1],
    [1, -1, -1, 1],
    [1, 1, -1, -1]
]) / 2

M2 = np.array([
    [1, -1, -i, -i],
    [1, 1, -i, i],
    [1, 1, i, -i],
    [1, -1, i, i]
]) / 2

M3 = np.array([
    [1, -i, -i, -1],
    [1, i, i, -1],
    [1, -i, i, 1],
    [1, i, -i, 1]
]) / 2

M4 = np.array([
    [1, -i, -1, -i],
    [1, i, -1, i],
    [1, -i, 1, i],
    [1, i, 1, -i]
]) / 2

# Function to create projectors from basis vectors
def create_projectors(basis):
    projectors = []
    for vector in basis.T:  # Transpose to access columns (vectors)
        projector = np.outer(vector, np.conj(vector))
        projectors.append(projector)
    return projectors

# Collect all projectors
all_projectors = []
for M in [M0, M1, M2, M3, M4]:
    all_projectors.extend(create_projectors(M))
