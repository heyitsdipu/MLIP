# src/gpr_pes/plotting.py

from matplotlib import pyplot as plt

def plot_contour(X, Y, Z, ax, title, colorscale="rainbow", levels=None):
    if levels is None:
        levels = [-12, -8, -4, 0, 4, 8, 10]
    ct = ax.contour(X, Y, Z, levels, colors='k')
    ax.clabel(ct, inline=True, fmt="%3.0f", fontsize=8)
    ct = ax.contourf(X, Y, Z, levels, cmap=colorscale, extend="both", vmin=levels[0], vmax=levels[-1])
    ax.set_xlabel("x", labelpad=-0.75)
    ax.set_ylabel("y", labelpad=2.5)
    cbar=plt.colorbar(ct)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=8)
