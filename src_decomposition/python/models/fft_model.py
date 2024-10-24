import torch
import torch.nn as nn

from vecopsciml.utils import TensOps
from vecopsciml.operators.zero_order import Mx, My

class FFTNeuralNetwork(nn.Module):

    def __init__(self, input_size, predictive_output_size, explanatory_output_size, **kwargs):
        super(FFTNeuralNetwork, self).__init__()

        self.input = input_size
        self.output_pred = predictive_output_size
        self.output_expl = explanatory_output_size

        self.hidden_units_pred = 10
        self.hidden_units_exp = 15
        self.filters_exp = 10

        # Predictive network
        self.flatten_layer_pred = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden1_layer_pred = nn.Linear(torch.prod(torch.tensor(self.input)), self.hidden_units_pred)  
        self.hidden2_layer_pred = nn.Linear(self.hidden_units_pred, self.hidden_units_pred)  
        self.output_layer_pred = nn.Linear(self.hidden_units_pred, self.output_pred[0]*self.output_pred[1])

        # Explanatory network
        self.conv1_exp = nn.Conv2d(in_channels=1, out_channels=self.filters_exp, kernel_size=1)
        self.flatten_layer_exp = nn.Flatten()
        self.hidden1_layer_exp = nn.LazyLinear(self.hidden_units_exp)
        self.hidden2_layer_exp = nn.Linear(self.hidden_units_exp, self.hidden_units_exp)
        self.output_layer_exp = nn.Linear(self.hidden_units_exp, self.filters_exp * (self.output_expl[0] - 1) * (self.output_expl[1] - 1))
        self.conv2_exp = nn.Conv2d(in_channels=self.filters_exp, out_channels=1, kernel_size=1)


    def forward(self, X):

        # Predictive network
        X = self.flatten_layer_pred(X)
        X = torch.sigmoid(self.hidden1_layer_pred(X))
        X = torch.sigmoid(self.hidden2_layer_pred(X))
        output_predictive_net = self.output_layer_pred(X)

        # Obtain real and imaginary part of output and transform it in a complex number
        output_predictive = output_predictive_net.view(output_predictive_net.size(0), self.output_pred[0], self.output_pred[1])
        real = output_predictive[..., 0]
        imag = output_predictive[..., 1]
        
        # Reconstruction of u(x, y) with iFFT
        f_coef = torch.zeros(output_predictive_net.size(0), 1, self.output_expl[0], self.output_expl[1], dtype=torch.complex64)
        f_coef[:, 0, 0:10, 0] = torch.complex(real, imag)

        u_pred = torch.fft.ifft2(f_coef).real
        um_pred = Mx(My(TensOps(u_pred, space_dimension=2, contravariance=0, covariance=0))).values

        # Explanatory network
        x = torch.sigmoid(self.conv1_exp(um_pred))
        x = self.flatten_layer_exp(x)
        x = torch.sigmoid(self.hidden1_layer_exp(x))
        x = torch.sigmoid(self.hidden2_layer_exp(x))
        x = self.output_layer_exp(x)
        x = x.view(x.size(0), self.filters_exp, self.output_expl[0] - 1, self.output_expl[1] - 1)
        K_pred = self.conv2_exp(x)

        return output_predictive, u_pred, K_pred