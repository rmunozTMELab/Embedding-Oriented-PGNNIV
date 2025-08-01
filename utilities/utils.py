import torch

class TensOps:
    """
    A class that represents a torch 'Tensor' and gives it mathematical context. In addition to the tensor values, 
    it also stores the space dimension in which the tensor exists (generally the set of real numbers),
    and its tensorial order (contravariant and covariant). Furthermore, internal operations between tensors 
    of the TensOps class are defined, such as addition, subtraction, negation, positive value, and equality checks.
    
    Attributes:
    ----------
    __values: torch.Tensor
        The tensor data represented as a PyTorch tensor.
    __space_dim: int
        The dimensionality of the space in which the tensor exists.
    __order: list[int]
        A list where the first element represents the contravariant order, 
        and the second element represents the covariant order of the tensor.
    """

    def __init__(self, data: torch.Tensor, space_dimension: int, contravariance: int, covariance: int):
        """
        Initialize the tensor object with tensor data, space dimensionality, and the contravariant/covariant order.

        Parameters:
        ----------
        data: torch.Tensor
            The tensor data.
        space_dimension: int
            The dimensionality of the space in which the tensor exists.
        contravariance: int
            The number of contravariant indices (upper indices).
        covariance: int
            The number of covariant indices (lower indices).
        """
        self.__values = data
        self.__space_dim = space_dimension
        self.__order = [contravariance, covariance]

    def check_dimensions(self):
        """
        Method placeholder for checking that the tensor's dimensions match 
        the space dimension and the expected contravariant/covariant order.
        """
        order = self.__order[0] + self.__order[1]
        space_dimension = self.__space_dim
        shape = self.__values.shape

        if not (len(shape[order:]) == space_dimension and len(shape[:space_dimension-1]) == order):
            raise ValueError("Tensorial order and space dimension don't match with the data.")

    @property
    def values(self):
        """
        Property to retrieve the tensor values (tensor data).
        """
        return self.__values
    
    @values.setter
    def values(self, new_data):
        """
        Setter for the tensor values, ensuring that the new data is a PyTorch tensor 
        and that it has the same shape as the current tensor.

        Parameters:
        ----------
        new_data: torch.Tensor
            The new tensor data to assign.

        Raises:
        -------
        ValueError:
            If the new data is not a PyTorch tensor or if its shape does not match the 
            existing tensor's shape.
        """
        if isinstance(new_data, torch.Tensor):
            if new_data.shape == self.__values.shape:
                self.__values = new_data
            else:
                raise ValueError(f"The shape of the new tensor {new_data.shape} does not match the expected shape {self.__values.shape}.")
        else:
            raise ValueError("The data must be a PyTorch tensor.")

    @property
    def space_dim(self):
        """
        Property to retrieve the space dimensionality of the tensor.
        """
        return self.__space_dim
    
    @property 
    def order(self):
        """
        Property to retrieve the tensor's contravariant and covariant order.
        """
        return self.__order
    
    def __add__(self, other):
        """
        Addition of two TensOps objects (element-wise tensor addition).
        
        Parameters:
        ----------
        other: TensOps
            Another TensOps object to add.
        
        Returns:
        -------
        TensOps:
            A new TensOps object with the result of the tensor addition.
        
        Raises:
        -------
        ValueError:
            If the tensors have different shapes.
        """
        if isinstance(other, TensOps) and self.__values.shape == other.values.shape:
            return TensOps(self.__values + other.values, self.__space_dim, *self.__order)
        else:
            raise ValueError("The tensors must have the same shape for addition.")

    def __sub__(self, other):
        """
        Subtraction of two TensOps objects (element-wise tensor subtraction).
        
        Parameters:
        ----------
        other: TensOps
            Another TensOps object to subtract.
        
        Returns:
        -------
        TensOps:
            A new TensOps object with the result of the tensor subtraction.
        
        Raises:
        -------
        ValueError:
            If the tensors have different shapes.
        """
        if isinstance(other, TensOps) and self.__values.shape == other.values.shape:
            return TensOps(self.__values - other.values, self.__space_dim, *self.__order)
        else:
            raise ValueError("The tensors must have the same shape for subtraction.")
        
    def __mul__(self, scalar):
        """
        Multiplication of the tensor by a scalar (element-wise).
        
        Parameters:
        ----------
        scalar: float or int
            The scalar value to multiply the tensor by.
        
        Returns:
        -------
        TensOps:
            A new TensOps object with the result of the scalar multiplication.
        """
        if isinstance(scalar, (int, float)):
            return TensOps(self.__values * scalar, self.__space_dim, *self.__order)
        else:
            raise ValueError("The scalar must be an integer or float.")
        
    def __truediv__(self, scalar):
        """
        Division of the tensor by a scalar (element-wise).
        
        Parameters:
        ----------
        scalar : float or int
            The scalar value to divide the tensor by.
        
        Returns:
        -------
        TensOps:
            A new TensOps object with the result of the scalar division.
        
        Raises:
        -------
        ValueError:
            If the scalar is not a number or if it is zero.
        """
        if isinstance(scalar, (int, float)):
            if scalar == 0:
                raise ValueError("Cannot divide by zero.")
            return TensOps(self.__values / scalar, self.__space_dim, *self.__order)
        else:
            raise ValueError("The scalar must be an integer or float.")

    def __neg__(self):
        """
        Negation of the tensor (element-wise).
        
        Returns:
        -------
        TensOps:
            A new TensOps object with the negated tensor values.
        """
        return TensOps(-self.__values, self.__space_dim, *self.__order)

    def __pos__(self):
        """
        Returns the tensor with positive values (element-wise).
        
        Returns:
        -------
        TensOps:
            A new TensOps object with the positive values of the tensor.
        """
        return TensOps(+self.__values, self.__space_dim, *self.__order)

    def __eq__(self, other):
        """
        Check if two TensOps objects are equal (element-wise comparison).
        
        Parameters:
        ----------
        other: TensOps
            Another TensOps object to compare with.
        
        Returns:
        -------
        bool:
            True if the tensors are equal (same values, same shape), False otherwise.
        """
        if isinstance(other, TensOps) and self.__values.shape == other.values.shape:
            return torch.equal(self.__values, other.values)
        return False

    def keys(self):
        """
        Return a list of the attributes (keys) of the tensor.
        
        Returns:
        --------
        list: A list of attribute names ['values', 'space_dim', 'contravariance', 'covariance'].
        """
        return ['values', 'space_dim', 'contravariance', 'covariance']

    def __str__(self):
        """
        Return a user-friendly string representation of the tensor object.
        This includes the type, shape, order (contravariant/covariant), 
        and space dimensionality of the tensor.

        Returns:
        -------
        str: A formatted string with tensor's details.
        """
        return (f"{type(self)}:\n"
                f" Shape: {list(self.__values.shape)}\n"
                f" Order: {self.__order}\n"
                f" Space dimension: {self.__space_dim}\n")

    def __repr__(self):
        """
        Return a detailed string representation for debugging purposes, 
        same as __str__ in this case.
        
        Returns:
        -------
        str : A string with the details of the tensor, useful for debugging.
        """
        return self.__str__()