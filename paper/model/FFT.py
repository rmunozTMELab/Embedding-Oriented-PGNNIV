import torch
import torch.nn as nn

from vecopsciml.utils import TensOps
from vecopsciml.operators.zero_order import Mx, My

def modes_base(data, n_modes):
    # FFT decomposition and obtain energy of each mode
    fft_data = torch.fft.fft2(data)
    energy = torch.abs(fft_data)
    energy_flattened = energy.flatten(1, 3)
    # Get the n_modes more energetic modes and their indices
    top_energetic = energy_flattened[:, :n_modes]
    # Create an empty template to include the modes
    filtered_modes = torch.zeros_like(energy, dtype=torch.complex64)
    filtered_modes.flatten(1, 3)[:, :n_modes] = fft_data.flatten(1, 3)[:, :n_modes]
    # Return the base with the 'n_modes' most energetic modes
    return filtered_modes

def reconstruct_data(coefficients_filtered):
    # Compute inverse FFT and reconstruct data
    reconstructed_data = torch.real(torch.fft.ifft2(coefficients_filtered))
    return reconstructed_data

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_layer_1_size, hidden_layer_2_size, latent_space_size):
        super(Encoder, self).__init__()

        # Parameters
        self.in_size = torch.tensor(input_size)
        self.h1_size = hidden_layer_1_size
        self.h2_size = hidden_layer_2_size
        self.ls_size = latent_space_size

        # Architecture
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden1_layer = nn.Linear(torch.prod(self.in_size), self.h1_size)
        self.hidden2_layer = nn.Linear(self.h1_size, self.h2_size)
        self.latent_space_layer = nn.Linear(self.h2_size, self.ls_size)
    
    def forward(self, X):
        
        X = self.flatten_layer(X)
        X = torch.sigmoid(self.hidden1_layer(X))
        X = torch.sigmoid(self.hidden2_layer(X))
        latent_space_output = (self.latent_space_layer(X))

        return latent_space_output
    
class Explanatory(nn.Module):

    def __init__(self, input_size, n_filters, hidden_layer_size, output_size):
        super(Explanatory, self).__init__()

        # Parameters
        self.in_size = torch.tensor(input_size)
        self.n_filters = n_filters * 1
        self.h_layer = hidden_layer_size
        self.out_size = torch.tensor(output_size)

        # Architecture
        self.conv_expand_layer = nn.Conv2d(in_channels=1, out_channels=self.n_filters, kernel_size=1)
        self.hidden_layer_1 = nn.Linear(self.n_filters, self.h_layer)
        self.hidden_layer_2 = nn.Linear(self.h_layer, self.n_filters)
        self.conv_converge_layer = nn.Conv2d(in_channels=self.n_filters, out_channels=1, kernel_size=1)
        
    def forward(self, X):

        batch_size, _, height, width = X.shape
        
        X = self.conv_expand_layer(X)
        X = X.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        X = X.reshape(-1, X.size(-1))  # (batch * height * width, channels)

        X = torch.sigmoid(self.hidden_layer_1(X))
        X = torch.sigmoid(self.hidden_layer_2(X))
        
        X = X.reshape(batch_size, height, width, -1)
        X = X.permute(0, 3, 1, 2)  # (batch, channels, height, width)
        explanatory_output = self.conv_converge_layer(X)

        return explanatory_output
    
class FFTNonlinearModel(nn.Module):
    
    def __init__(self, input_size, predictive_layers, FFT_modes_base, output_predictive_size, explanatory_input_size, explanatory_layers, output_explanatory_size, n_filters, device):
        
        super(FFTNonlinearModel, self).__init__()

        # Parameters
        self.in_size = input_size
        self.pred_size = predictive_layers
        self.out_pred_size = output_predictive_size
        
        self.in_exp_size = explanatory_input_size
        self.exp_size = explanatory_layers
        self.out_exp_size = output_explanatory_size

        self.n_filters = n_filters
        self.device = device

        # Architecture
        self.encoder = Encoder(self.in_size, self.pred_size[0], self.pred_size[1], 2*self.pred_size[2])
        self.base_indices = FFT_modes_base
        self.explanatory = Explanatory(self.in_exp_size, self.n_filters, self.exp_size[0], self.out_exp_size)
        
    def forward(self, X):

        # Predictive network
        X = self.encoder(X)

        # Manipulating output to obtain real and complex part
        output_predictive = X.view(X.size(0), self.pred_size[2], 2)
        real = output_predictive[..., 0]
        imag = output_predictive[..., 1]
    
        # Reconstruction with FFT and manipulation of prediction output
        base = torch.zeros((X.size(0), *self.out_pred_size), dtype=torch.complex64).to(self.device)
        base.flatten(1, 3)[:, :self.base_indices] = torch.complex(real, imag)

        u = reconstruct_data(base).to(self.device)        
        um = Mx(My(TensOps(u, space_dimension=2, contravariance=0, covariance=0))).values

        # Explanatory network
        K = self.explanatory(um)
        
        return u, K