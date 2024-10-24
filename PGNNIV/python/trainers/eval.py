import torch

from vecopsciml.algebra import zero_order as azo
from vecopsciml.operators import zero_order as zo
from vecopsciml.utils import TensOps

def e_constraint(y_true, y_pred):
    """
    Computes the difference between the true output and predicted output tensors (error constraint).

    Args:
    - y_true: Ground truth tensor.
    - y_pred: Predicted tensor.

    Returns:
    - Tensor representing the difference between `y_true` and `y_pred`.
    """
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    e = y_true - y_pred
    return e

def pi1_constraint(y_pred, K_pred, f_true, D):
    """
    Computes the first physical constraint related to the divergence of flux.

    Args:
    - y_pred: Predicted tensor (output).
    - K_pred: Predicted diffusivity tensor.
    - f_true: Ground truth forcing term tensor.
    - D: Grid spacing or related parameter.

    Returns:
    - Difference between the predicted and true forcing term tensors.
    """
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    K_pred = TensOps(K_pred, space_dimension=2, contravariance=0, covariance=0)

    qx = azo.scalar_product(K_pred, zo.Dx(y_pred, D))
    qy = azo.scalar_product(K_pred, zo.Dy(y_pred, D))

    dqxdx = zo.Dx(qx, D)
    dqydy = zo.Dy(qy, D)

    f_pred = -(dqxdx + dqydy)

    return f_pred - zo.Mx(zo.Mx(zo.My(zo.My(f_true))))

def pi2_constraint(X_true, y_pred):
    """
    Computes boundary constraints by comparing the predicted and true boundary values.

    Args:
    - X_true: Ground truth boundary tensor.
    - y_pred: Predicted output tensor.

    Returns:
    - Tensor representing the difference between predicted and true boundary values along the edges.
    """
    return torch.concat([
        y_pred[:, :, :, 0].unsqueeze(1) - X_true[:, :, :, 0].unsqueeze(1),
        y_pred[:, :, :, -1].unsqueeze(1) - X_true[:, :, :, 1].unsqueeze(1),
        y_pred[:, :, 0, :].unsqueeze(1) - X_true[:, :, :, 2].unsqueeze(1),
        y_pred[:, :, -1, :].unsqueeze(1) - X_true[:, :, :, 3].unsqueeze(1)
    ], dim=1)

def pi3_constraint(X_true, y_pred, K_pred, D):
    """
    Computes the boundary flux constraint, comparing the predicted fluxes with the true fluxes at the boundaries.

    Args:
    - X_true: Ground truth tensor.
    - y_pred: Predicted tensor (output).
    - K_pred: Predicted diffusivity tensor.
    - D: Grid spacing or related parameter.

    Returns:
    - Tensor representing the difference between the predicted and true boundary fluxes.
    """
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    K_pred = TensOps(K_pred, space_dimension=2, contravariance=0, covariance=0)

    qx_pred = azo.scalar_product(K_pred, zo.Dx(y_pred, D)).values
    qy_pred = azo.scalar_product(K_pred, zo.Dy(y_pred, D)).values

    X_true = TensOps(X_true, space_dimension=2, contravariance=0, covariance=0)
    X_true_red = zo.My(X_true).values

    return torch.concat([
        qx_pred[:, :, :, 0].unsqueeze(1) - X_true_red[:, :, :, 4].unsqueeze(1),
        qx_pred[:, :, :, -1].unsqueeze(1) - X_true_red[:, :, :, 5].unsqueeze(1),
        qy_pred[:, :, 0, :].unsqueeze(1) - X_true_red[:, :, :, 6].unsqueeze(1),
        qy_pred[:, :, -1, :].unsqueeze(1) - X_true_red[:, :, :, 7].unsqueeze(1)
    ], dim=1)

def MSE(diff_tensor):
    """
    Computes the mean squared error for a given tensor, summing the squares of all its components.

    Args:
    - diff_tensor: Tensor or TensOps object for which to compute MSE.

    Returns:
    - Scalar value representing the sum of squared differences.
    """
    if isinstance(diff_tensor, torch.Tensor):
        return torch.sum(torch.square(diff_tensor))
    else:
        return torch.sum(torch.square(diff_tensor.values))

def loss_function(X_true, y_true, y_pred, K_pred, f_true, D):
    """
    Computes the total loss as a weighted sum of the errors for various constraints (e, pi1, pi2, pi3).

    Args:
    - X_true: Ground truth boundary tensor.
    - y_true: Ground truth output tensor.
    - y_pred: Predicted output tensor.
    - K_pred: Predicted diffusivity tensor.
    - f_true: Ground truth forcing term tensor.
    - D: Grid spacing or related parameter.

    Returns:
    - total_loss: Scalar representing the total loss.
    - e, pi1, pi2, pi3: Individual loss components for each constraint.
    """
    e = MSE(e_constraint(y_true, y_pred))
    pi1 = MSE(pi1_constraint(y_pred, K_pred, f_true, D))
    pi2 = MSE(pi2_constraint(X_true, y_pred))
    pi3 = MSE(pi3_constraint(X_true, y_pred, K_pred, D))

    total_loss = 1e7 * e + 1e4 * pi1 + 1e3 * pi2 + 1e5 * pi3

    return total_loss, e, pi1, pi2, pi3