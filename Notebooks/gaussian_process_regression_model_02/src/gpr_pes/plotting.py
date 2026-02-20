# src/gpr_pes/plotting.py

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.axes import Axes

def plot_contour(X: np.ndarray, 
                 Y: np.ndarray, 
                 Z: np.ndarray,
                 ax: Axes, 
                 title: str, 
                 colorscale: str = "rainbow", 
                 levels=None):
    """
    Plot filler contour map with contour lines
    """
    if levels is None:
        levels = [-12, -8, -4, 0, 4, 8, 10]

    # Line contours
    ct = ax.contour(X, Y, Z, levels, colors='k')
    ax.clabel(ct, inline=True, fmt="%3.0f", fontsize=8)

    # Filled contours
    filled_ct = ax.contourf(X, Y, Z, levels, cmap=colorscale, extend="both", vmin=levels[0], vmax=levels[-1])
    
    ax.set_xlabel("x", labelpad=-0.75)
    ax.set_ylabel("y", labelpad=2.5)
    ax.set_title(title, fontsize=8)

    # Colorbar linked to axis
    cbar=plt.colorbar(filled_ct)
    cbar.ax.tick_params(labelsize=8)
    
    return filled_ct

