import torch

from vecopsciml.algebra import zero_order as azo
from vecopsciml.operators import zero_order as zo
from vecopsciml.utils import TensOps

# Constraint e
def e_constraint(y_true, y_pred):

    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    e = y_true - y_pred
    return e

def pi1_constraint(y_pred, K_pred, f_true, D):

    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    K_pred = TensOps(K_pred, space_dimension=2, contravariance=0, covariance=0)

    qx = azo.scalar_product(K_pred, zo.Dx(y_pred, D))
    qy = azo.scalar_product(K_pred, zo.Dy(y_pred, D))

    dqxdx = zo.Dx(qx, D)
    dqydy = zo.Dy(qy, D)

    f_pred = -(dqxdx + dqydy)

    return f_pred - zo.Mx(zo.Mx(zo.My(zo.My(f_true))))


# Constraint pi2
def pi2_constraint(X_true, y_pred):

    return torch.concat([y_pred[:, :, :, 0].unsqueeze(1) - X_true[:, :, :, 0].unsqueeze(1),
                         y_pred[:, :, :, -1].unsqueeze(1) - X_true[:, :, :, 1].unsqueeze(1), 
                         y_pred[:, :, 0, :].unsqueeze(1) - X_true[:, :, :, 2].unsqueeze(1),
                         y_pred[:, :, -1, :].unsqueeze(1) - X_true[:, :, :, 3].unsqueeze(1),
                         ], dim=1)

# Constraint pi3
def pi3_constraint(X_true, y_pred, K_pred, D):
    
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    K_pred = TensOps(K_pred, space_dimension=2, contravariance=0, covariance=0)

    qx_pred = azo.scalar_product(K_pred, zo.Dx(y_pred, D)).values
    qy_pred = azo.scalar_product(K_pred, zo.Dy(y_pred, D)).values

    X_true = TensOps(X_true, space_dimension=2, contravariance=0, covariance=0)
    X_true_red = (zo.My(X_true)).values

    return torch.concat([qx_pred[:, :, :, 0].unsqueeze(1) - X_true_red[:, :, :, 4].unsqueeze(1),
                         qx_pred[:, :, :, -1].unsqueeze(1) - X_true_red[:, :, :, 5].unsqueeze(1),
                         qy_pred[:, :, 0, :].unsqueeze(1) - X_true_red[:, :, :, 6].unsqueeze(1),
                         qy_pred[:, :, -1, :].unsqueeze(1) - X_true_red[:, :, :, 7].unsqueeze(1),
                         ], dim=1)


# MSE as defined in the paper: sum of all the squarers of all the components of a tensor
def MSE(diff_tensor):
    if isinstance(diff_tensor, torch.Tensor):
        return torch.sum(torch.square(diff_tensor))
    else:
        return torch.sum(torch.square(diff_tensor.values))
    

def loss_function(X_true, y_true, y_pred, u_pred, K_pred, f_true, D):

    e = MSE(e_constraint(y_true, y_pred))
    pi1 = MSE(pi1_constraint(u_pred, K_pred, f_true, D))
    pi2 = MSE(pi2_constraint(X_true, u_pred))
    pi3 = MSE(pi3_constraint(X_true, u_pred, K_pred, D))

    total_loss = 1e8*e + 1e4*pi1 + 1e3*pi2 + 1e5*pi3

    return total_loss, e, pi1, pi2, pi3