import torch.nn as nn
import torch

from vecopsciml.algebra import zero_order as azo
from vecopsciml.operators import zero_order as zo
from vecopsciml.kernels.derivative import DerivativeKernels
from vecopsciml.utils import Tensor


def pi1_constraint(y_pred, K_pred, D):

    y_pred = Tensor(y_pred, space_dimension=2, order=0)
    K_pred = Tensor(K_pred, space_dimension=2, order=0)

    dy = zo.gradient(y_pred, D)

    qx = Tensor(-torch.mul(K_pred.values, dy.values[:, 0].unsqueeze(1)), space_dimension=2, order=0)
    qy = Tensor(-torch.mul(K_pred.values, dy.values[:, 1].unsqueeze(1)), space_dimension=2, order=0)

    dqxdx = zo.gradient(qx, D).values[:, 0].unsqueeze(1)
    dqydy = zo.gradient(qy, D).values[:, 1].unsqueeze(1)

    pi1 = dqxdx + dqydy

    return Tensor(pi1, space_dimension=2, order=0)
    # return pi1


# Constraint e
def e_constraint(y_true, y_pred):

    y_pred = Tensor(y_pred, y_true.space_dim, y_true.order)
    e = azo.subtraction(y_true, y_pred)
    return e


# Constraint pi2
def pi2_constraint(X_true, y_pred):

    X_true = X_true.values
    y_pred = y_pred

    return torch.concat([y_pred[:, :, :, 0].unsqueeze(1) - X_true[:, :, :, 0].unsqueeze(1),
                         y_pred[:, :, :, -1].unsqueeze(1) - X_true[:, :, :, 1].unsqueeze(1), 
                         y_pred[:, :, 0].unsqueeze(1) - X_true[:, :, :, 2].unsqueeze(1),
                         y_pred[:, :, -1].unsqueeze(1) - X_true[:, :, :, 3].unsqueeze(1),
                         ], dim=1)

# Constraint pi3
def pi3_constraint(X_true, y_pred, K_pred, D):
    
    y_pred = Tensor(y_pred, space_dimension=2, order=0)
    K_pred = Tensor(K_pred, space_dimension=2, order=0)

    dy = zo.gradient(y_pred, D)

    qx_pred = -torch.mul(K_pred.values, dy.values[:, 0].unsqueeze(1)) 
    qy_pred = -torch.mul(K_pred.values, dy.values[:, 1].unsqueeze(1))

    X_true_red = (zo.My(X_true)).values

    return torch.concat([qx_pred[:, :, :, 0].unsqueeze(1) - X_true_red[:, :, :, 4].unsqueeze(1),
                         qx_pred[:, :, :, -1].unsqueeze(1) - X_true_red[:, :, :, 5].unsqueeze(1),
                         qy_pred[:, :, :, 0].unsqueeze(1) - X_true_red[:, :, :, 6].unsqueeze(1),
                         qy_pred[:, :, :, -1].unsqueeze(1) - X_true_red[:, :, :, 7].unsqueeze(1),
                         ], dim=1)


# MSE as defined in the paper: sum of all the squarers of all the components of a tensor
def MSE(diff_tensor):
    if isinstance(diff_tensor, torch.Tensor):
        return torch.sum(torch.square(diff_tensor))
    else:
        return torch.sum(torch.square(diff_tensor.values))
    

def loss_function(X_true, y_true, y_pred, K_pred, D):

    e = MSE(e_constraint(y_true, y_pred))
    pi1 = MSE(pi1_constraint(y_pred, K_pred, D))
    pi2 = MSE(pi2_constraint(X_true, y_pred))
    pi3 = MSE(pi3_constraint(X_true, y_pred, K_pred, D))

    total_loss = 1e7*e + 1e4*pi1 + 1e3*pi2 + 1e5*pi3

    return total_loss, e, pi1, pi2, pi3