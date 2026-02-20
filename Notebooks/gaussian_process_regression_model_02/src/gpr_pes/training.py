# src/gpr_pes/train.py
import torch
import gpytorch

def train_model(model, likelihood, x_train, y_train, print_hp=False):
    """
    Train Gaussian Process hyperparameters by maximizing 
    the marginal log likelihood.

    Parameters:
    -----------
    model : ExactGPModel
        GP model defining prior mean and covariance

    likelihood : GaussianLikelihood
        Observation noise model.

    x_train : torch.Tensor
        Training input locations (N x D)

    y_train : torch.Tensor
        Training outputs (N, )

    print_hp: bool
    """
    # Initialize hyperparameters to reasonable starting values
    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(1.0),
        'covar_module.base_kernel.lengthscale': torch.tensor(0.5),
        'covar_module.outputscale': torch.tensor(0.5),
    }
    model.initialize(**hypers)

    if print_hp:
        # Print initial hyperparmeters
        print("--------------------------------------------------------------------")
        for param_name, param in model.named_parameters():
            print(f"Parmeters name: {param_name:42} value = {param.item():9.5f}")
        print("--------------------------------------------------------------------")

    training_iter = 1000 # Number of gradient steps

    # Optimizer (using stochastic gradient descent)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Marginial log likelihood (MLL)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):

        # clear gradients from previous iteration
        optimizer.zero_grad()

        # Forward pass:
        output = model(x_train)

        # Compute negative log marginal likelihood
        loss = -mll(output, y_train)

        # Backpropagation
        loss.backward()

        # monitoring
        if (i+1) % 10 == 0 and print_hp:
            print(f"Iter: {i+1:3d} | Loss: {loss.item():.3f} | Output scale: {model.covar_module.outputscale.item():.3f} | Length scale: {model.covar_module.base_kernel.lengthscale.item():.3f} | Noise: {model.likelihood.noise.item():.3f} ")

        # Update hyperparameters
        optimizer.step()
    print("--------------------------------------------------------------------")
    if print_hp:
        # Print optimized hyperparameters
        for param_name, param in model.named_parameters():
            print(f"Parameter name: {param_name:42} value = {param.item():9.5f}")
    print("--------------------------------------------------------------------")