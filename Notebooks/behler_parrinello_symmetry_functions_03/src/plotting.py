import matplotlib.pyplot as plt
import numpy as np

def cutoff_function(Rij, Rc):
    """
    Compute the cutoff function f_c(R_ij).

    Parameters
    ----------
    Rij : array-like
        Pairwise distances.

    Rc : float
        Cutoff distance.

    Returns
    -------
    fc : array-like
        Cutoff function values for each R_ij.
    """
    Rij = np.asarray(Rij)
    fc = 0.5 * (np.cos(np.pi * Rij / Rc) + 1)
    fc[Rij > Rc] = 0.0
    return fc



def plot_pairwise_vector(Rij, Rc, etas):
    """
    Plot radial symmetry function G(R_ij) for multiple eta values.

    Parameters
    ----------
    Rij : array-like
        Pairwise distances.

    Rc : float
        Cutoff distance.

    etas : array-like
        List or array of eta values controlling Gaussian width.

    Notes
    -----
    Radial symmetry function:

        G(R_ij) = exp[-η (R_ij - R_c)^2] f_c(R_ij)

    Cutoff function:

        f_c(R_ij) = 0.5 [cos(π R_ij / R_c) + 1]   if R_ij ≤ R_c
                    0                            otherwise
    """

    Rij = np.asarray(Rij)

    # Cutoff function with proper truncation
    fc = cutoff_function(Rij, Rc)

    plt.figure(figsize=(8, 6))

    # Plot curves for different eta values
    for eta in etas:
        G = np.exp(-eta * (Rij - Rc)**2) * fc
        plt.plot(Rij, G, alpha=1.0, lw=3, label=rf"$\eta = {eta}$")

    plt.title("Radial Symmetry Function")
    plt.xlabel(r"$R_{\text{ij}}$", fontsize=14)
    plt.ylabel(r"$G(R_{\text{ij}})$", fontsize=14)
    plt.tick_params(axis='both', which='major',
                direction='in', top=True, right=True,
                length=6, width=1)

    plt.tick_params(axis='both', which='minor',
                direction='in', top=True, right=True,
                length=3, width=1)
    plt.legend(loc='upper right', fontsize=14)
    plt.xlim(0, Rc)
    plt.ylim(0, 1.05)

    plt.show()