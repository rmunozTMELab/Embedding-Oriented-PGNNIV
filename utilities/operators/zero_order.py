import numpy as np
import torch
import torch.nn.functional as func
from utilities.utils import TensOps

from utilities.kernels.average import AverageKernels

def Mx(f):
    """
    Computes the average in x direction of a second-order tensor.

    Parameters:
    - f (TensOps): An instance of the TensOps class that represents the matricial field. Must be a tensor of order 2. It must have shape = [N, 2//3, 2//3, nx, ny, (nz)].
    
    Returns:
    - average (TensOps): the average of the field in one direction of 'f'.
    """
    # Check if 'f' is a TensOps instance
    if not isinstance(f, TensOps):
        raise TypeError(f"Argument 'f' must be of type TensOps (class TensOps), not {type(f).__name__}.")
    
    # Check tensorial order of 'f'
    if np.sum(f.order) != 0:
        raise ValueError("'f' must be a first-order tensor.")
    
    f_ = f.values

    if f.space_dim == 2:
        M = (torch.tensor([[+1, +1]], dtype=torch.float32)/(2)).unsqueeze(0).unsqueeze(0)
        average = func.conv2d(f_, M, stride=1, padding='valid')
        return TensOps(average, space_dimension=2, contravariance=0, covariance=0)
    
    if f.space_dim == 3:
        raise ValueError("Not implemented yet. Under development.")
    
    raise ValueError(
        f"TensOps values must have 2 or 3 as space dimension.\n"
        f"Received dim of 'f': {f.space_dim}.")

def My(f):
    """
    Computes the average in y direction of a second-order tensor.

    Parameters:
    - f (TensOps): An instance of the TensOps class that represents the matricial field. Must be a tensor of order 2. It must have shape = [N, 2//3, 2//3, nx, ny, (nz)].
    
    Returns:
    - average (TensOps): the average of the field in one direction of 'f'.
    """
    # Check if 'f' is a TensOps instance
    if not isinstance(f, TensOps):
        raise TypeError(f"Argument 'f' must be of type TensOps (class TensOps), not {type(f).__name__}.")
    
    # Check tensorial order of 'f'
    if np.sum(f.order) != 0:
        raise ValueError("'f' must be a first-order tensor.")
    
    f_ = f.values

    if f.space_dim == 2:
        M = (torch.tensor([[+1], [+1]], dtype=torch.float32)/(2)).unsqueeze(0).unsqueeze(0)
        average = func.conv2d(f_, M, stride=1, padding='valid')
        return TensOps(average, space_dimension=2, contravariance=0, covariance=0)
    
    if f.space_dim == 3:
        raise ValueError("Not implemented yet. Under development.")
    
    raise ValueError(
        f"TensOps values must have 2 or 3 as space dimension.\n"
        f"Received dim of 'f': {f.space_dim}.")


def Dx(f, D):

    # Check if 'f' is a TensOps instance
    if not isinstance(f, TensOps):
        raise TypeError(f"Argument 'f' must be of type TensOps (class TensOps), not {type(f).__name__}.")
    
    # Check tensorial order of 'f'
    if np.sum(f.order) != 0:
        raise ValueError("'f' must be a zero-order tensor.")
    
    f_ = f.values
    
    # Case in which 'f' is a scalar field in two dimensions. 
    if f.space_dim == 2:
        if D.shape[0] != 2:
            raise ValueError("Field dimensions and number of filters don't match")
        
        Dx = D[0]

        # Compute the gradient
        dfdx = func.conv2d(f_, Dx, stride=1, padding='valid')
        
        return TensOps(dfdx, space_dimension=2, contravariance=0, covariance=0)
        
    # Case in which 'f' is a scalar field in three dimensions. 
    if f.space_dim == 3:
        if D.shape[0] != 3:
            raise ValueError("Field dimensions and number of filters don't match")

        Dx = D[0]

        # Compute the gradient
        dfdx = func.conv3d(f_, Dx, stride=1, padding='valid')

        return TensOps(dfdx, space_dimension=3, contravariance=0, covariance=0)

    raise ValueError(
        f"TensOps values must have 2 or 3 as space dimension.\n"
        f"Received dim of 'f': {f.space_dim}.")


def Dy(f, D):

    # Check if 'f' is a TensOps instance
    if not isinstance(f, TensOps):
        raise TypeError(f"Argument 'f' must be of type TensOps (class TensOps), not {type(f).__name__}.")
    
    # Check tensorial order of 'f'
    if np.sum(f.order) != 0:
        raise ValueError("'f' must be a zero-order tensor.")
    
    f_ = f.values
    
    # Case in which 'f' is a scalar field in two dimensions. 
    if f.space_dim == 2:
        if D.shape[0] != 2:
            raise ValueError("Field dimensions and number of filters don't match")
        
        Dy = D[1]

        # Compute the gradient
        dfdy = func.conv2d(f_, Dy, stride=1, padding='valid')
        
        return TensOps(dfdy, space_dimension=2, contravariance=0, covariance=0)
        
    # Case in which 'f' is a scalar field in three dimensions. 
    if f.space_dim == 3:
        if D.shape[0] != 3:
            raise ValueError("Field dimensions and number of filters don't match")

        Dy = D[1]

        # Compute the gradient
        dfdy = func.conv3d(f_, Dx, stride=1, padding='valid')

        return TensOps(dfdy, space_dimension=3, contravariance=0, covariance=0)

    raise ValueError(
        f"TensOps values must have 2 or 3 as space dimension.\n"
        f"Received dim of 'f': {f.space_dim}.")