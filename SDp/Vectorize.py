import sympy as sp

# Define the matrix A
A = sp.Matrix([
    [3, 1, -1, 1],
    [1, 3, 1, -1],
    [-1, 1, 3, 1],
    [1, -1, 1, 3]
]) / 4

# Compute the eigenvalues and their algebraic and geometric multiplicities
eigen_info = A.eigenvals()

# Create the matrix P and the Jordan form Gamma
P, J = A.jordan_form()

# Output the results
print("Matrix P (transformation matrix):")
sp.pprint(P, use_unicode=True)

print("\nJordan Canonical Form (Gamma):")
sp.pprint(J, use_unicode=True)
