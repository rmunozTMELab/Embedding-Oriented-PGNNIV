import torch

from vecopsciml.algebra import zero_order as azo
from vecopsciml.operators import zero_order as zo
from vecopsciml.utils import TensOps


def e_constraint(y_true, y_pred):
    """
    Computes the constraint 'e' as the difference between the true values and predicted values.

    Parameters:
        y_true (TensOps): The ground truth values of the solution.
        y_pred (torch.Tensor): The predicted solution of the model.

    Returns:
        e (TensOps): The difference tensor, representing the error or constraint.
    """
    
    # Convert y_pred into a 'TensOps' object
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)

    # Calculate the difference between the true values and predictions.
    e = y_true - y_pred
    
    return e


def pi1_constraint(y_pred, K_pred, f_true, D):
    """
    Computes a constraint that compares the predicted source term with the ground truth.

    Parameters:
        y_pred (torch.Tensor): The predicted solution of the model.
        K_pred (torch.Tensor): The predicted diffusivity of the model.
        f_true (TensOps): Ground truth source term of the equation.
        D (torch.Tensor): Derivative filters.

    Returns:
        The difference between the predicted and ground truth result of the heat equation.
    """

    # Convert y_pred and K_pred into a 'TensOps' objects.
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    K_pred = TensOps(K_pred, space_dimension=2, contravariance=0, covariance=0)

    # Compute the predicted flow in x and y direction
    qx = azo.scalar_product(K_pred, zo.Dx(y_pred, D))
    qy = azo.scalar_product(K_pred, zo.Dy(y_pred, D))

    # Compute the predicted heat equation
    dqxdx = zo.Dx(qx, D)
    dqydy = zo.Dy(qy, D)
    f_pred = -(dqxdx + dqydy)

    # Return the difference between the ground truth and the prediction
    return f_pred - zo.Mx(zo.Mx(zo.My(zo.My(f_true))))


def pi2_constraint(X_true, y_pred):
    """
    Computes a constraint that obtains the difference between the predicted and true boundary conditions.

    Parameters:
        X_true (torch.Tensor): True boundary conditions for the system. They are the input of the model.
        y_pred (torch.Tensor): Predicted solution for the entire domain.

    Returns:
       (torch.Tensor): Differences between predicted and true boundary values evaluated along the edges of the domain.
    """

    # Returns the difference between the boundary values of the prediction y_pred and corresponding true values.
    return torch.concat([y_pred[:, :, :, 0].unsqueeze(1) - X_true[:, :, :, 0].unsqueeze(1),
                         y_pred[:, :, :, -1].unsqueeze(1) - X_true[:, :, :, 1].unsqueeze(1), 
                         y_pred[:, :, 0, :].unsqueeze(1) - X_true[:, :, :, 2].unsqueeze(1),
                         y_pred[:, :, -1, :].unsqueeze(1) - X_true[:, :, :, 3].unsqueeze(1),
                         ], dim=1)


def pi3_constraint(X_true, y_pred, K_pred, D):
    """
    Computes a constraint that obtains the difference between predicted and true fluxes at the boundaries of the domain.

    Parameters:
        X_true (torch.Tensor): True boundary conditions for the system. They are the input of the model.
        y_pred (torch.Tensor): Predicted solution for the entire domain.
        K_pred (torch.Tensor): Predicted diffusivity for the entire domain.
        D (torch.Tensor): Derivative filters.

    Returns:
        (torch.Tensor): Differences between predicted and true flow at the boundaries of the domain.
    """
    
    # Convert y_pred and K_pred into a 'TensOps' objects.
    y_pred = TensOps(y_pred, space_dimension=2, contravariance=0, covariance=0)
    K_pred = TensOps(K_pred, space_dimension=2, contravariance=0, covariance=0)

    # Compute the predicted flow in x and y direction.
    qx_pred = azo.scalar_product(K_pred, zo.Dx(y_pred, D)).values
    qy_pred = azo.scalar_product(K_pred, zo.Dy(y_pred, D)).values

    # Convert X_true into a TensOps to apply the average operator to obtain the values in the elements instead of in the nodes.
    X_true = TensOps(X_true, space_dimension=2, contravariance=0, covariance=0)
    X_true_red = (zo.My(X_true)).values

    # Compute differences between predicted fluxes and reduced true flux values at boundaries:
    return torch.concat([qx_pred[:, :, :, 0].unsqueeze(1) - X_true_red[:, :, :, 4].unsqueeze(1),
                         qx_pred[:, :, :, -1].unsqueeze(1) - X_true_red[:, :, :, 5].unsqueeze(1),
                         qy_pred[:, :, 0, :].unsqueeze(1) - X_true_red[:, :, :, 6].unsqueeze(1),
                         qy_pred[:, :, -1, :].unsqueeze(1) - X_true_red[:, :, :, 7].unsqueeze(1),
                         ], dim=1)


def MSE(diff_tensor):
    """
    Computes the sum of the squares of all components in the given tensor.

    Parameters:
        diff_tensor (torch.Tensor or TensOps): Object containing the difference values.

    Returns:
        (torch.Tensor): The MSE value, defined as the sum of squared components in the tensor.

    Raises:
        ValueError: An error occurred accessing the smalltable.
    """

     # For regular PyTorch tensors, directly compute the sum of squares.
    if isinstance(diff_tensor, torch.Tensor):
        return torch.sum(torch.square(diff_tensor))
    
    # For TensOps objects, access the '.values' and compute the sum of squares.
    elif isinstance(diff_tensor, TensOps):
        return torch.sum(torch.square(diff_tensor.values))
    
    # In other case, raise a ValueError
    else:
        raise ValueError("Not compatible data type. Need torch.Tensor or TensOps object.")
   

def loss_function(X_true, y_true, y_pred, K_pred, f_true, D):
    """
    Computes the total loss as a weighted sum of constraints (e, pi1, pi2, pi3).

    Parameters:
        X_true (torch.Tensor): True boundary conditions for the system. They are the input of the model.
        y_true (TensOps): The ground truth values of the solution.
        y_pred (torch.Tensor): Predicted solution for the entire domain.
        K_pred (torch.Tensor): Predicted diffusivity for the entire domain.
        f_true (TensOps): Ground truth source term of the equation.
        D (torch.Tensor): Derivative filters.

    Returns:
        total_loss (float): The total loss, computed as a weighted sum of individual constraint losses.
        e (float): Loss contribution from the `e_constraint`.
        pi1 (float): Loss contribution from the `pi1_constraint`.
        pi2 (float): Loss contribution from the `pi2_constraint`.
        pi3 (float): Loss contribution from the `pi3_constraint`.
    """
    
    # Compute the MSE for each individual constraint
    e = MSE(e_constraint(y_true, y_pred))  
    pi1 = MSE(pi1_constraint(y_pred, K_pred, f_true, D))  
    pi2 = MSE(pi2_constraint(X_true, y_pred))  
    pi3 = MSE(pi3_constraint(X_true, y_pred, K_pred, D))

    # Combine individual losses into a total loss with specific weighting factors.
    total_loss = 1e7 * e + 1e4 * pi1 + 1e3 * pi2 + 1e5 * pi3

    # Return the total loss and individual components.
    return total_loss, e, pi1, pi2, pi3