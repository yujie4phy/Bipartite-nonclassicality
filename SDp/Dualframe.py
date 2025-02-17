import numpy as np


def is_full_rank(matrix):
    """Check if the given matrix has full rank."""
    return np.linalg.matrix_rank(matrix) == min(matrix.shape)


def verify_reconstruction(f_vectors, g_vectors, test_vectors):
    """Verify the reconstruction property for the test vectors."""
    F = np.column_stack(f_vectors)
    G = np.column_stack(g_vectors)

    for test_vec in test_vectors:
        reconstructed_vec = sum(np.dot(test_vec, g) * f for g, f in zip(g_vectors, f_vectors))
        if not np.allclose(test_vec, reconstructed_vec):
            return False
    return True


# Define the frame vectors f_a
f_vectors = [
    np.array([1 / 6, np.sqrt(3) / 18 * np.cos(np.pi / 3), np.sqrt(3) / 18 * np.sin(np.pi / 3)]),
    np.array([1 / 6, np.sqrt(3) / 18 * np.cos(2 * np.pi / 3), np.sqrt(3) / 18 * np.sin(2 * np.pi / 3)]),
    np.array([1 / 6, np.sqrt(3) / 18 * np.cos(np.pi), np.sqrt(3) / 18 * np.sin(np.pi)]),
    np.array([1 / 6, np.sqrt(3) / 18 * np.cos(4 * np.pi / 3), np.sqrt(3) / 18 * np.sin(4 * np.pi / 3)]),
    np.array([1 / 6, np.sqrt(3) / 18 * np.cos(5 * np.pi / 3), np.sqrt(3) / 18 * np.sin(5 * np.pi / 3)]),
    np.array([1 / 6, np.sqrt(3) / 18 * np.cos(2 * np.pi), np.sqrt(3) / 18 * np.sin(2 * np.pi)])
]

# Define the new frame vectors g_a (candidate dual frame)
g_vectors = [
    np.array([1 / 6, 2 / 6 * np.sin(np.pi / 3), 2 / 6 * np.cos(np.pi / 3)]),
    np.array([1 / 6, 2 / 6 * np.sin(2 * np.pi / 3), 2 / 6 * np.cos(2 * np.pi / 3)]),
    np.array([1 / 6, 2 / 6 * np.sin(np.pi), 2 / 6 * np.cos(np.pi)]),
    np.array([1 / 6, 2 / 6 * np.sin(4 * np.pi / 3), 2 / 6 * np.cos(4 * np.pi / 3)]),
    np.array([1 / 6, 2 / 6 * np.sin(5 * np.pi / 3), 2 / 6 * np.cos(5 * np.pi/ 3)]),
    np.array([1 / 6, 2 / 6 * np.sin(2 * np.pi), 2 / 6 * np.cos(2 * np.pi)])
]

# Compute G^T F
G = np.column_stack(g_vectors)/2
F = np.column_stack(f_vectors)
G_T_F = G.T @ F

# Check if G^T F has full rank
full_rank = is_full_rank(G_T_F)

# Define some test vectors (these should be representative of the space)
test_vectors = [
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 1]),
    np.array([1, 1, 1])
]

# Verify the reconstruction property
reconstruction = verify_reconstruction(f_vectors, g_vectors, test_vectors)

full_rank, reconstruction
