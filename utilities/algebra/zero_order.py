import numpy as np
import torch
from utilities.utils import TensOps

def scalar_product(a:TensOps, b:TensOps):
    """
    Calculates the product of two tensors that represent two scalar fields.

    Args:
        a (TensOps): An instance of the TensOps class representing a scalar field.
        b (TensOps): An instance of the TensOps class representing a scalar field.

    Returns:
        TensOps: A TensOps instance containing the product of 'a' and 'b'.
    """
    # Check if 'a' and 'b' are TensOps instances
    if not isinstance(a, TensOps):
        raise TypeError(f"Argument 'a' must be of type TensOps, not {type(a).__name__}.")
    if not isinstance(b, TensOps):
        raise TypeError(f"Argument 'b' must be of type TensOps, not {type(b).__name__}.")

    # Check tensorial order of 'a' and 'b'
    if np.sum(a.order) != 0:
        raise ValueError("'a' must be a zero-order TensOps.")
    if np.sum(b.order) != 0:
        raise ValueError("'b' must be a zero-order TensOps.")

    a_ = a.values
    b_ = b.values

    # Check if tensors values have the same shape and are compatible
    if a_.shape != b_.shape:
        raise ValueError(
            f"TensOps values must have the same shape.\n"
            f"Received shape of a: {a_.shape}.\n"
            f"Received shape of b: {b_.shape}."
        )

    # Compute the dot product
    if a.space_dim == 2 and b.space_dim == 2:
        scalar_prod = a_ * b_
        return TensOps(scalar_prod, space_dimension=2, contravariance=0, covariance=0)

    if a.space_dim == 3 and b.space_dim == 3:
        scalar_prod = a_ * b_
        return TensOps(scalar_prod, space_dimension=3, contravariance=0, covariance=0)

    raise ValueError(
        f"TensOps values must have the same space dimension.\n"
        f"Received dim of 'a': {a.space_dim}.\n"
        f"Received dim of 'b': {b.space_dim}."
    )