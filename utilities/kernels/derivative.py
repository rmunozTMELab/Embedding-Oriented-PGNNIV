import torch

class DerivativeKernels:

    def __init__(self, dx, dy, dz):
        '''
        Initializes the DerivativeKernels object.
        This class is used to create the filters (or kernels) that are necessary to compute numerical derivatives in 2D and 3D spaces using PyTorch.
        They compute central differences method.

        Parameters:
        dx (float): Grid spacing in the x direction.
        dy (float): Grid spacing in the y direction.
        dz (float): Grid spacing in the z direction.
        '''
        self.dx = dx
        self.dy = dy
        self.dz = dz

    def grad_kernels_two_dimensions(self):
        '''
        Computes gradient kernels for images or 2D data, allowing you to approximate the first derivative along the x and y directions.

        Returns:
        torch.Tensor: A tensor containing the two kernels stacked with shape = [2, 1, 1, 2, 2] according to https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        '''
        Dx = torch.tensor([[-1, +1], 
                           [-1, +1]], 
                          dtype=torch.float32)/(2*self.dx)
        
        Dy = torch.tensor([[-1, -1], 
                           [+1, +1]], 
                          dtype=torch.float32)/(2*self.dy)
   
        D_kernel = torch.stack([
            Dx.unsqueeze(0).unsqueeze(0),  
            Dy.unsqueeze(0).unsqueeze(0),
        ], dim=0) 
        
        return D_kernel

    def grad_kernels_three_dimensions(self):
        '''
        Computes gradient kernels for volumetric or 3D data, allowing you to approximate the first derivative along the x, y, and z directions.

        Returns:
        torch.Tensor: A tensor containing the three kernels stacked with shape = [3, 1, 1, 2, 2, 2] according to https://pytorch.org/docs/stable/generated/torch.nn.functional.conv3d.html

        '''
        Dx = torch.tensor(
            [
                [[-1, -1], [+1, +1]],
                [[-1, -1], [+1, +1]]
            ],
        dtype=torch.float32)/(4*self.dx)

        Dy = torch.tensor(
            [
                [[-1, -1], [-1, -1]],
                [[+1, +1], [+1, +1]]
            ],
        dtype=torch.float32)/(4*self.dy)
        
        Dz = torch.tensor(
            [
                [[-1, +1], [-1, +1]],
                [[-1, +1], [-1, +1]]
            ],
        dtype=torch.float32)/(4*self.dz)
        
        D_kernel = torch.stack([
            Dx.unsqueeze(0).unsqueeze(0),  
            Dy.unsqueeze(0).unsqueeze(0),
            Dz.unsqueeze(0).unsqueeze(0),
        ], dim=0)   
        
        return D_kernel
