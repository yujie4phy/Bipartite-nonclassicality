import numpy as np
from scipy.spatial import ConvexHull

phi = (1 + np.sqrt(5)) / 2  # Golden ratio
icosahedron_vertices = np.array([
    [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
    [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
    [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
])

# Create a convex hull to find the triangular faces
hull = ConvexHull(icosahedron_vertices)
faces = hull.simplices  # This gives an array of indices for the vertices forming each triangular face

# Now calculate the centroids for the dodecahedron vertices
dodecahedron_vertices = np.array([np.mean(icosahedron_vertices[face], axis=0) for face in faces])

# Normalize these centroids
dodecahedron_vertices_normalized = dodecahedron_vertices / np.linalg.norm(dodecahedron_vertices, axis=1)[:, np.newaxis]

# Print results to check
print("Dodecahedron Vertices (normalized):")
print(dodecahedron_vertices_normalized)