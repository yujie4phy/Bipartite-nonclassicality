import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
### Code for plotting Fig 5: Experimental violation in the main text

# 1. Data
p_vals = np.array([
    0.290, 0.442, 0.430, 0.454, 0.465, 0.483,
    0.505, 0.534, 0.551, 0.651, 0.742, 0.985
])
p_errs = np.array([
    0.002, 0.002, 0.003, 0.002, 0.002, 0.003,
    0.003, 0.003, 0.002, 0.003, 0.004, 0.004
])
I_vals = np.array([
     0.04513,  0.00578,  0.01068,  -0.00117, -0.00409, -0.00585,
    -0.01207, -0.01633, -0.02583, -0.05280, -0.07849, -0.14256
])
I_errs = np.array([
    0.00015, 0.00021, 0.00019, 0.00022, 0.00018, 0.00039,
    0.00018, 0.00016, 0.00022, 0.00011, 0.00012, 0.00013
])

# 2. Linear fit
m, b = np.polyfit(p_vals, I_vals, 1)
x_fit = np.linspace(np.min(p_vals), np.max(p_vals), 200)
y_fit = m * x_fit + b

# 3. Ideal function
C0 = 0.118034
C1 = -0.281389
y_ideal = C0 + C1 * x_fit

fig, ax = plt.subplots(figsize=(7, 5))

# Plot data with error bars
ax.errorbar(
    p_vals, I_vals,
    xerr=p_errs, yerr=I_errs,
    fmt='*', markersize=4, color='brown', ecolor='darkred', capsize=3
)

# Plot the linear fit and the ideal function
ax.plot(x_fit, y_fit, color='powderblue', linewidth=3)
ax.plot(x_fit, y_ideal, color='dimgrey', linestyle='--', linewidth=2)

# Add vertical dashed line and gray shading
ax.axvline(x=0.5, color='lightcoral', linestyle='--', linewidth=2)
ax.axhspan(0, 0.08, color='grey', alpha=0.15)

ax.set_xlabel(r'$p$', fontsize=12)
ax.set_ylabel(r'$\mathcal{I}$', fontsize=12)
ax.set_xlim(0.28, 1.0)
ax.set_ylim(-0.15, 0.05)
ax.yaxis.set_major_locator(mticker.MultipleLocator(0.05))

# Do NOT call ax.legend() to remove the legend

# 4. Inset
axins = inset_axes(ax, width="20%", height="20%", loc='upper right', borderpad=2)

axins.errorbar(
    p_vals, I_vals,
    xerr=p_errs, yerr=I_errs,
    fmt='*', color='brown', ecolor='darkred', capsize=2
)
axins.plot(x_fit, y_fit, color='powderblue', linewidth=3)
axins.plot(x_fit, y_ideal, color='dimgrey', linestyle='--', linewidth=2)
axins.axvline(x=0.5, color='lightcoral', linestyle='--', linewidth=2)
axins.axhspan(-0.01, 0.0, color='grey', alpha=0.15)

# Set the zoom region in the inset
axins.set_xlim(0.4, 0.52)
axins.set_ylim(-0.01, 0.01)

# Restrict the tick values for the inset
axins.set_xticks([0.4, 0.45,0.50])
axins.set_yticks([-0.01, 0.00,0.01])
highlight_idx = [3, 4]
# ax.scatter(p_vals[highlight_idx], I_vals[highlight_idx],
#            s=100, marker='*', facecolor='yellow', edgecolor='black', zorder=5)
axins.scatter(p_vals[highlight_idx], I_vals[highlight_idx],
            s=100, marker='*', facecolor='yellow', edgecolor='black', zorder=5)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.tight_layout()
plt.show()