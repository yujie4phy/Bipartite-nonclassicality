import numpy as np
from decimal import Decimal, getcontext

# Set precision for Decimal calculations
getcontext().prec = 10

def Planor(k):
    XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
    ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
    A = np.zeros((k, 2,2), dtype=complex)
    for i in range(k):
        theta = (i) * 2*np.pi / k
        A[i, :, :] = 1/k * (np.eye(2) +0.58*(np.sin(theta) * XX + np.cos(theta) * ZZ))
    return A

def Planortilde(k):
    XX = np.array([[0, 1], [1, 0]])  # Pauli X matrix
    ZZ = np.array([[1, 0], [0, -1]])  # Pauli Z matrix
    A = np.zeros((k, 2,2), dtype=complex)
    for i in range(k):
        theta = (i) * 2*np.pi / k+np.pi/k
        A[i, :, :] = 1/k * (np.eye(2) + (np.sin(theta) * XX + np.cos(theta) * ZZ))
    return A

def MUB2():
    A = np.zeros((6, 2, 2), dtype=complex)
    A[0, :, :] = np.array([[1, 0], [0, 0]]);
    A[1, :, :] = np.array([[0, 0], [0, 1]]);
    A[2, :, :] = 1 / 2 * np.array([[1, 1], [1, 1]]);
    A[3, :, :] = 1 / 2 * np.array([[1, -1], [-1, 1]]);
    A[4, :, :] = 1 / 2 * np.array([[1, 1j], [-1j, 1]]);
    A[5, :, :] = 1 / 2 * np.array([[1, -1j], [1j, 1]]);
    return A

def MUB4():
    M0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    M1 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, -1, -1, 1], [1, 1, -1, -1]]) / 2
    M2 = np.array([[1, -1, -1j, -1j], [1, 1, -1j, 1j], [1, 1, 1j, -1j], [1, -1, 1j, 1j]]) / 2
    M3 = np.array([[1, -1j, -1j, -1], [1, 1j, 1j, -1], [1, -1j, 1j, 1], [1, 1j, -1j, 1]]) / 2
    M4 = np.array([[1, -1j, -1, -1j], [1, 1j, -1, 1j], [1, -1j, 1, 1j], [1, 1j, 1, -1j]]) / 2
    def create_projectors(basis):
        projectors = []
        for vector in basis:  # Transpose to access columns (vectors)
            projector = np.outer(vector, np.conj(vector))
            projectors.append(projector)
        return projectors
    A = []
    for M in [M0, M1,M2,M3,M4]:
        A.extend(create_projectors(M))
    A=np.array(A)/5
    return A

def create_povm(vertices):
    povm = []
    for v in vertices:
        bloch_vector = np.array(v)
        # Normalize the Bloch vector to ensure it lies on the Bloch sphere
        bloch_vector = bloch_vector / np.linalg.norm(bloch_vector)
        # Convert Bloch vector to density matrix
        rho = 0.5 * (np.eye(2) + np.array([
            [bloch_vector[2], bloch_vector[0] - 1j * bloch_vector[1]],
            [bloch_vector[0] + 1j * bloch_vector[1], -bloch_vector[2]]
        ]))
        povm.append(rho)
    return 2*np.array(povm)/len(vertices)

def create_povmT(vertices):
    povm = []
    for v in vertices:
        bloch_vector = np.array(v)
        # Normalize the Bloch vector to ensure it lies on the Bloch sphere
        bloch_vector = bloch_vector / np.linalg.norm(bloch_vector)
        # Convert Bloch vector to density matrix
        rho = 0.5 * (np.eye(2) + np.array([
            [bloch_vector[2], bloch_vector[0] + 1j * bloch_vector[1]],
            [bloch_vector[0] - 1j * bloch_vector[1], -bloch_vector[2]]
        ]))
        povm.append(rho)
    return 2*np.array(povm)/len(vertices)

def tetrahedron_povm():
    vertices = [
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1]
    ]
    return create_povm(vertices)

def cube_povm():
    vertices = [
        [1, 1, 1],
        [1, 1, -1],
        [1, -1, 1],
        [1, -1, -1],
        [-1, 1, 1],
        [-1, 1, -1],
        [-1, -1, 1],
        [-1, -1, -1]
    ]
    return create_povm(vertices)

def octahedron_povm():
    vertices = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1]
    ]
    return create_povm(vertices)

def dodecahedron_povm():
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices=[[-1,-1,-1],[-phi,1/phi,0],[-phi,-1/phi,0],[-1,-1,1],
       [0,-phi,1/phi],[1,-1,-1], [0,-phi,-1/phi],[-1,1,-1],
       [-1/phi,0,-phi],[1/phi,0,-phi],[0,phi,-1/phi],[1,1,-1],
       [0, phi,1/phi],[-1/phi,0,phi],[-1,1,1],[phi,-1/phi,0],
       [phi,1/phi,0],[1,-1,1], [1,1,1],[1/phi,0,phi]]
    return create_povm(vertices)

def icosahedron_povm():
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    return create_povm(vertices)
def iscodode_POVM():
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    vertices = [
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1 / phi, phi], [0, 1 / phi, -phi], [0, -1 / phi, phi], [0, -1 / phi, -phi],
        [1 / phi, phi, 0], [1 / phi, -phi, 0], [-1 / phi, phi, 0], [-1 / phi, -phi, 0],
        [phi, 0, 1 / phi], [phi, 0, -1 / phi], [-phi, 0, 1 / phi], [-phi, 0, -1 / phi]
    ]
    return create_povm(vertices)

def UPB_POVM():
    vertices = [
        [1/3, 2/3, 0],
        [1/3, -2/3, 0],
        [2/3, 1/3, 0],
        [2/3, -2/3, 0],
        [-1/3, 2/3, 0],
        [-1/3, -2/3, 0],
        [-2/3, 1/3, 0],
        [-2/3, -2/3, 0],
    ]
    return create_povm(vertices)