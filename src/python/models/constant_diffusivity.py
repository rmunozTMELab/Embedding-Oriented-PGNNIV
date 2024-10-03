import torch
import torch.nn as nn
import torch.nn.functional as F

class ConstantDiffusivityNeuralNetwork(nn.Module):

    def __init__(self, input_size, hidden1_dim, hidden2_dim, output_size):
        super(ConstantDiffusivityNeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.output_size = output_size

        # Predictive network
        self.flatten_layer_pred = nn.Flatten()
        self.hidden1_layer_pred = nn.Linear(torch.prod(torch.tensor(self.input_size)), self.hidden1_dim)  # Ajusta las dimensiones de entrada y salida
        self.hidden2_layer_pred = nn.Linear(self.hidden1_dim, self.hidden2_dim)  # Ajusta según el tamaño de la salida de hidden1
        self.output_layer_pred = nn.Linear(self.hidden2_dim, self.output_size[1] * self.output_size[2])  # Ajusta según el tamaño de la salida de hidden2

        # Matrix of trainable parameters
        self.weight_matrix = nn.Parameter(torch.randn(output_size[0], output_size[1] - 1, output_size[2] - 1))

    def forward(self, X):

        # Predictive network
        X_pred_flat = self.flatten_layer_pred(X)
        X_pred_hidden1 = torch.sigmoid(self.hidden1_layer_pred(X_pred_flat))
        X_pred_hidden2 = torch.sigmoid(self.hidden2_layer_pred(X_pred_hidden1))
        output_dense_pred = torch.sigmoid(self.output_layer_pred(X_pred_hidden2))

        # Outputs 
        u_pred = output_dense_pred.view(-1, 1, self.output_size[1], self.output_size[2])
        K_pred = self.weight_matrix

        return u_pred, K_pred