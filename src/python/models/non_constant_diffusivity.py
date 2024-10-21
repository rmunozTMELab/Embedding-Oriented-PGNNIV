import torch
import torch.nn as nn

from vecopsciml.utils import TensOps
from vecopsciml.operators.zero_order import Mx, My

class NonConstantDiffusivityNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_dim, hidden2_dim, output_size, n_filters=5, **kwargs):
        super(NonConstantDiffusivityNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_size = output_size
        self.n_filters = n_filters

        # Predictive network
        self.flatten_layer_pred = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden1_layer_pred = nn.Linear(torch.prod(torch.tensor(self.input_size)), self.hidden1_dim)  
        self.hidden2_layer_pred = nn.Linear(self.hidden1_dim, self.hidden2_dim)  
        self.output_layer_pred = nn.Linear(self.hidden2_dim, self.output_size[2] * self.output_size[2])  

        # Explanatory network
        self.conv1_exp = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=1)
        self.flatten_layer_exp = nn.Flatten()
        self.hidden1_layer_exp = nn.LazyLinear(hidden1_dim)
        self.hidden2_layer_exp = nn.Linear(hidden1_dim, hidden2_dim)
        self.output_layer_exp = nn.Linear(hidden2_dim, n_filters * (output_size[1] - 1) * (output_size[2] - 1))
        self.conv2_exp = nn.Conv2d(in_channels=n_filters, out_channels=1, kernel_size=1)
    
    def forward(self, X):

        # Predictive network
        X_pred_flat = self.flatten_layer_pred(X)
        X_pred_hidden1 = torch.sigmoid(self.hidden1_layer_pred(X_pred_flat))
        X_pred_hidden2 = torch.sigmoid(self.hidden2_layer_pred(X_pred_hidden1))
        output_dense_pred = self.output_layer_pred(X_pred_hidden2)

        # Mean operator
        u_pred = output_dense_pred.view(output_dense_pred.size(0), 1, self.output_size[1], self.output_size[2])
        um_pred = Mx(My(TensOps(u_pred, space_dimension=2, contravariance=0, covariance=0))).values

        
        # Explanatory network
        x = torch.sigmoid(self.conv1_exp(um_pred))
        x = self.flatten_layer_exp(x)
        x = torch.sigmoid(self.hidden1_layer_exp(x))
        x = torch.sigmoid(self.hidden2_layer_exp(x))
        x = self.output_layer_exp(x)
        x = x.view(x.size(0), self.n_filters, self.output_size[1] - 1, self.output_size[2] - 1)
        K_pred = self.conv2_exp(x)

        return u_pred, K_pred

