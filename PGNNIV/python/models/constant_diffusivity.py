import torch
import torch.nn as nn

class ConstantDiffusivityNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_dim, hidden2_dim, output_size):
        """
        Initializes the neural network model. The architecture consists of an input layer, two hidden layers, and an output layer.
        Additionally, a trainable weight matrix is included as part of the model. In terms of the heat equation, this will be
        the diffusivity in the case of an homogeneous or heterogeneous problem.
        
        Args:
        - input_size: Size of the input tensor.
        - hidden1_dim: The number of neurons in the first hidden layer.
        - hidden2_dim: The number of neurons in the second hidden layer.
        - output_size: The shape of the output tensor.
        """
        super(ConstantDiffusivityNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_size = output_size

        # Predictive network
        self.flatten_layer_pred = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden1_layer_pred = nn.Linear(torch.prod(torch.tensor(self.input_size)), self.hidden1_dim)  
        self.hidden2_layer_pred = nn.Linear(self.hidden1_dim, self.hidden2_dim)  
        self.output_layer_pred = nn.Linear(self.hidden2_dim, self.output_size[1] * self.output_size[2])  

        # Matrix of trainable parameters
        self.weight_matrix = nn.Parameter(torch.randn(output_size[0], output_size[1] - 1, output_size[2] - 1))

    def forward(self, X):
        """
        Forward pass of the neural network. This function defines how input data passes through the layers.
        
        Args:
        - X: Input tensor.
        
        Returns:
        - u_pred: Output of the predictive network reshaped to the desired output dimensions.
        - K_pred: The learned weight matrix which serves as a constant diffusivity factor.
        """
        # Predictive network
        X_pred_flat = self.flatten_layer_pred(X)
        X_pred_hidden1 = torch.sigmoid(self.hidden1_layer_pred(X_pred_flat))
        X_pred_hidden2 = torch.sigmoid(self.hidden2_layer_pred(X_pred_hidden1))
        output_dense_pred = torch.sigmoid(self.output_layer_pred(X_pred_hidden2))
        u_pred = output_dense_pred.view(-1, 1, self.output_size[1], self.output_size[2])

        # Explanatory weight matrix
        K_pred = self.weight_matrix

        return u_pred, K_pred