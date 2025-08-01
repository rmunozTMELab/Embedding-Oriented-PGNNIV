import torch

class AverageKernels:

    def __init__(self):
        '''
        Initializes the AverageKernels object.
        This class is used to create the filters (or kernels) that are necessary to compute the average of a grid in 2D and 3D spaces using PyTorch.
        They compute central differences method.
        '''

    def average_kernels_two_dimensions(self):
        '''
        Computes gradient kernels for images or 2D data, allowing you to approximate the first derivative along the x and y directions.

        Returns:
        torch.Tensor: A tensor containing the two kernels stacked with shape = [2, 1, 1, 2, 2] according to https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        '''
        Mx = torch.tensor([[+1, +1]],
                         dtype=torch.float32)/(2)
        
        My = torch.tensor([[+1], 
                             [+1]],
                         dtype=torch.float32)/(2)
        
        print(My.shape)
        
        D_kernel = torch.stack([
            Mx.unsqueeze(0).unsqueeze(0),  
            My.unsqueeze(0).unsqueeze(0),
        ], dim=0) 
        
        return D_kernel

    def average_kernels_three_dimensions(self):
        '''
        Computes gradient kernels for volumetric or 3D data, allowing you to approximate the first derivative along the x, y, and z directions.

        Returns:
        torch.Tensor: A tensor containing the three kernels stacked with shape = [3, 1, 1, 2, 2, 2] according to https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html

        '''
        raise ValueError("Average kernels not created for 3D yet. Under developement.")
