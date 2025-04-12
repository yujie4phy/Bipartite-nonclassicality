import numpy as np

def rotation_matrix(theta):
    """Return a 2×2 rotation matrix for an angle theta (radians)."""
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def jones_waveplate(angle, delta):
    """
    Return the Jones matrix for a waveplate rotated by 'angle' (radians)
    with retardance 'delta' (radians).
    J(angle, delta) = R(-angle) · diag(1, exp(i*delta)) · R(angle)
    """
    R = rotation_matrix(angle)
    R_inv = rotation_matrix(-angle)
    return R_inv @ np.diag([1, np.exp(1j * delta)]) @ R

def generate_noisy_jones_state(hwp_angle, qwp_angle,
                               hwp_rot_noise_rad, qwp_rot_noise_rad,
                               hwp_ret_noise_rad, qwp_ret_noise_rad):
    """
    Generate a noisy Jones state using:
      - Ideal angles: hwp_angle, qwp_angle   (radians)
      - Rotation noise added: hwp_rot_noise_rad, qwp_rot_noise_rad (radians)
      - Retardance noise added to the ideal π (HWP) and π/2 (QWP).

    The final angles are:
      HWP rotation = hwp_angle + hwp_rot_noise_rad
      HWP retardance = π + hwp_ret_noise_rad
      QWP rotation = qwp_angle + qwp_rot_noise_rad
      QWP retardance = π/2 + qwp_ret_noise_rad
    """
    # Noisy rotation angles
    hwp_noisy_angle = hwp_angle + hwp_rot_noise_rad
    qwp_noisy_angle = qwp_angle + qwp_rot_noise_rad

    # Noisy retardances
    hwp_noisy_delta = np.pi + hwp_ret_noise_rad
    qwp_noisy_delta = (np.pi / 2) + qwp_ret_noise_rad

    # Construct the waveplate Jones matrices
    J_hwp = jones_waveplate(hwp_noisy_angle, hwp_noisy_delta)
    J_qwp = jones_waveplate(qwp_noisy_angle, qwp_noisy_delta)

    # Input state = horizontal polarization |H> = [1, 0]^T
    psi_noisy = J_qwp @ J_hwp @ np.array([1, 0], dtype=complex)
    psi_noisy /= np.linalg.norm(psi_noisy)
    return psi_noisy

def bloch_from_state(psi):
    """
    Convert a pure state (Jones vector) |psi> to its Bloch vector representation.
    v_i = 2 * Re{ <psi| σ_i |psi> }
    """
    rho = np.outer(psi, np.conjugate(psi))
    pauli_X = np.array([[0, 1], [1, 0]])
    pauli_Y = np.array([[0, -1j], [1j, 0]])
    pauli_Z = np.array([[1, 0], [0, -1]])

    vx = np.real(np.trace(rho @ pauli_X))
    vy = np.real(np.trace(rho @ pauli_Y))
    vz = np.real(np.trace(rho @ pauli_Z))
    return np.array([vx, vy, vz])

def effect_from_bloch(v):
    """
    Given a Bloch vector v = (v_x, v_y, v_z), return the corresponding 2×2 effect:
        E = 0.5 * (I + v_x σ_x + v_y σ_y + v_z σ_z)
    (weight = 1).
    """
    I = np.eye(2)
    pauli_X = np.array([[0, 1], [1, 0]])
    pauli_Y = np.array([[0, -1j], [1j, 0]])
    pauli_Z = np.array([[1, 0], [0, -1]])
    return 0.5 * (I + v[0] * pauli_X + v[1] * pauli_Y + v[2] * pauli_Z)

def perturbation(
    angle_pairs,      # shape (N, 2) array of [hwp_angle, qwp_angle]
    pair_list,        # list of tuples: each (i, j) indexes a pair
    hwp_rot_std_deg=1,
    qwp_rot_std_deg=1,
    hwp_ret_std_deg=1,
    qwp_ret_std_deg=1
):
    """
    For a given array of waveplate angles (angle_pairs) and a pairing map (pair_list),
    apply the SAME random noise to both items in each pair. That is, for each pair (i, j):
      1) Generate a single (hwp_rot_noise, qwp_rot_noise, hwp_ret_noise, qwp_ret_noise).
      2) Apply exactly that noise to angle_pairs[i] and angle_pairs[j].
      3) Construct the corresponding 2x2 effects.

    Any angle index not included in the pair_list is left unperturbed (or you can
    choose to generate noise for them too if that is desired—just adapt the code).

    Returns:
      A list (or array) of 2x2 perturbed effects, in the same index order as angle_pairs.
    """
    N = len(angle_pairs)
    # Initialize array for storing the 2x2 effects after perturbation
    perturbed_effects = [None] * N

    # Keep track of which indices are in pairs
    used_indices = set()
    for i, j in pair_list:
        used_indices.add(i)
        used_indices.add(j)

    # Step 1: handle paired indices
    for (i, j) in pair_list:
        # Sample ONE random noise set for the entire pair
        hwp_rot_noise_rad = np.deg2rad(np.random.normal(0, hwp_rot_std_deg))
        qwp_rot_noise_rad = np.deg2rad(np.random.normal(0, qwp_rot_std_deg))
        hwp_ret_noise_rad = np.deg2rad(np.random.normal(0, hwp_ret_std_deg))
        qwp_ret_noise_rad = np.deg2rad(np.random.normal(0, qwp_ret_std_deg))

        # Apply the same noise to effect i
        hwp_i, qwp_i = angle_pairs[i]
        psi_i = generate_noisy_jones_state(
            hwp_i, qwp_i,
            hwp_rot_noise_rad, qwp_rot_noise_rad,
            hwp_ret_noise_rad, qwp_ret_noise_rad
        )
        v_i = bloch_from_state(psi_i)
        E_i = effect_from_bloch(v_i)
        perturbed_effects[i] = E_i

        # Apply the same noise to effect j
        hwp_j, qwp_j = angle_pairs[j]
        psi_j = generate_noisy_jones_state(
            hwp_j, qwp_j,
            hwp_rot_noise_rad, qwp_rot_noise_rad,
            hwp_ret_noise_rad, qwp_ret_noise_rad
        )
        v_j = bloch_from_state(psi_j)
        E_j = effect_from_bloch(v_j)
        perturbed_effects[j] = E_j

    return np.array(perturbed_effects)

# ------------------------------
# Example usage
# ------------------------------