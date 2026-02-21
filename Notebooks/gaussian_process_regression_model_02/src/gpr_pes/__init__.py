# src/gpr_pes/__init__.py
from .model import ExactGPModel, GPModelWithDerivatives 
from .plotting import plot_contour
from .training import train_model, train_model_with_derivatives

__all__ = ["ExactGPModel",
           "GPModelWithDerivatives"
           "plot_contour",
           "train_model",
           "train_model_with_derivatives"]
