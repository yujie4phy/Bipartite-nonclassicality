import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_rho_3d(ax, rho_fitted, opt_rho, title=None):
    """
    Plots the real parts of two 4x4 density matrices (rho_fitted and opt_rho)
    as 3D bars on the given Axes3D 'ax'.
    """
    assert rho_fitted.shape == opt_rho.shape, "rho_fitted and opt_rho must have the same shape."
    N = rho_fitted.shape[0]
    if N != 4:
        raise ValueError("This function is set up for 4x4 matrices.")

    rf = np.real(rho_fitted)
    ro = np.real(opt_rho)

    # Determine global min/max for colormap
    all_vals = np.concatenate([rf.flatten(), ro.flatten()])
    vmin, vmax = np.min(all_vals), np.max(all_vals)

    # Choose a diverging colormap
    cmap = plt.get_cmap('RdYlGn')
    dx = dy = 0.5

    # Plot rho_fitted (semi-opaque)
    for i in range(N):
        for j in range(N):
            z_val = rf[i, j]
            color = cmap((z_val - vmin)/(vmax - vmin) if vmax != vmin else 0.5)
            ax.bar3d(i, j, 0, dx, dy, z_val, color=color, edgecolor='k', alpha=0.5)

    # Overlay opt_rho (more transparent)
    for i in range(N):
        for j in range(N):
            z_val = ro[i, j]
            color = cmap((z_val - vmin)/(vmax - vmin) if vmax != vmin else 0.5)
            ax.bar3d(i, j, 0, dx, dy, z_val, color=color, edgecolor='k', alpha=0.2)

    # Set ticks at centers
    tick_positions = [0.25, 1.25, 2.25, 3.25]
    tick_labels = ["HH", "HV", "VH", "VV"]
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticklabels(tick_labels, fontsize=10)

    # We reduce label font sizes and add minimal padding

    # Set the subplot title, if given
    if title:
        ax.set_title(title, pad=6, fontsize=13)

    ax.set_box_aspect((1,1,1))
    ax.view_init(elev=20, azim=-45)

# ---------------------------------------------------------------------
# Example: Plot 12 subplots in a 3×4 grid, with minimal overlap.
# Load your results array (previously saved) with 12 entries:
results = np.load("results-mento.npy", allow_pickle=True).tolist()

# Make a bigger figure and manually adjust spacing.
#   - figsize=(16, 9) is just an example; increase if needed.
#   - wspace=0.05 reduces horizontal spacing.
#   - top=0.88 leaves more room for titles on the top row.
fig = plt.figure(figsize=(30, 9))
fig.subplots_adjust(
    left=0.0,   # space on left edge
    right=1,  # space on right edge
    top=0.88,    # space at top
    bottom=0.05, # space at bottom
    wspace=0, # horizontal space between subplots
    hspace=0.25  # vertical space between subplots
)
punl=[0.002,0.002,0.003,0.002,0.002,0.003,0.003,0.003,0.002,0.003,0.004,0.004]
for idx, entry in enumerate(results[:12], start=1):
    ax = fig.add_subplot(3, 4, idx, projection='3d')
    # Title using your parameter (r_in or p).
    # Example: p=0.27
    p_val = entry['optr']  # or rename if it's 'p'
    p_unc=punl[idx-1]
    title_str = f"p = {p_val:.3f} ± {p_unc:.3f}"

    # Plot
    plot_rho_3d(ax, entry['rho_fitted'], entry['opt_rho'], title=title_str)

    # Optional: remove repeated axis labels
    # Only bottom row shows X tick labels
    if idx not in [9, 10, 11, 12]:  # if not in bottom row
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    # # Only left column shows Y tick labels
    # if idx not in [1, 5, 9]:
    #     ax.set_yticklabels([])
plt.show()