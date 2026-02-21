# src/gpr_pes/model.py
"""
Gaussian Process Regression models for potential energy surface (PES) fitting.

This module contains the ExactGPModel used with GPyTorch for regression on
low-dimensional PES data (e.g., Z(x, y)).
"""

from __future__ import annotations

import gpytorch
import torch


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact Gaussian Process regression model.

    This model defines a probabilistic prior over functions:
        f(x) ~ GP(mean(x), covariance(x, x'))

    In GPs we do not learn weights like in neural networks. We learn kernel
    hyperparameters (e.g., lengthscale, outputscale, noise) by maximizing
    the marginal likelihood of the observed data.

    Parameters
    ----------
    train_x : torch.Tensor
        Training inputs with shape (N, D).
        Example: D=2 for a PES depending on (x, y).

    train_y : torch.Tensor
        Training targets with shape (N,) (or (N, 1) if you standardize to that).
        Example: potential values Z(x, y).

    likelihood : gpytorch.likelihoods.GaussianLikelihood
        Gaussian observation model:
            y = f(x) + ε,  ε ~ N(0, σ_n²)
        Noise helps handle measurement/model mismatch and prevents forced
        exact interpolation.
    """

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood: gpytorch.likelihoods.GaussianLikelihood,
    ) -> None:
        super().__init__(train_x, train_y, likelihood)

        # Mean function: prior expectation (0 by default).
        self.mean_module = gpytorch.means.ZeroMean()

        # Covariance function (kernel):
        # RBF kernel assumes smoothness:
        #   k(x, x') = σ_f² exp( -||x-x'||² / (2 l²) )
        # ScaleKernel wraps the base kernel to learn σ_f² (outputscale).
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Return the GP prior/posterior (depending on train/eval mode) at inputs x.

        Parameters
        ----------
        x : torch.Tensor
            Query inputs with shape (M, D).

        Returns
        -------
        gpytorch.distributions.MultivariateNormal
            Multivariate normal distribution over f(x).
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    """"
    
    Exact Gaussian Regression Model with gradient
    
    """
    def __init__(
        self, 
        train_x: torch.tensor, 
        train_y: torch.tensor, 
        likelihood: gpytorch.likelihoods.GaussianLikelihood, 
    ) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        
        # Covariance function (kernel):
        # RBF kernel assumes smoothness:
        #   k(x, x') = σ_f² exp( -||x-x'||² / (2 l²) )
        # ScaleKernel wraps the base kernel to learn σ_f² (outputscale).
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        
        
