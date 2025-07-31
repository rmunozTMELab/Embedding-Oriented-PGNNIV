import torch
import torch.nn as nn

from vecopsciml.utils import TensOps
from vecopsciml.operators.zero_order import Mx, My

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
    
class Decoder(nn.Module):

    def __init__(self, latent_space_size, hidden_layer_1_size, hidden_layer_2_size, output_size):
        super(Decoder, self).__init__()

        # Parameters
        self.ls_size = latent_space_size
        self.h1_size = hidden_layer_1_size
        self.h2_size = hidden_layer_2_size
        self.out_size = torch.tensor(output_size)

        # Architecture
        self.hidden1_layer = nn.Linear(self.ls_size, self.h1_size)
        self.hidden2_layer = nn.Linear(self.h1_size, self.h2_size)
        self.output_layer = nn.Linear(self.h2_size, torch.prod(self.out_size))
    
    def forward(self, X):
        
        X = torch.sigmoid(self.hidden1_layer(X))
        X = torch.sigmoid(self.hidden2_layer(X))
        decoder_output = self.output_layer(X)

        return decoder_output
    
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
    
class PGNNIVBaseline(nn.Module):
    
    def __init__(self, input_size, predictive_layers, output_predictive_size, explanatory_input_size, explanatory_layers, output_explanatory_size, n_filters):
        
        super(PGNNIVBaseline, self).__init__()

        # Parameters
        self.in_size = input_size
        self.pred_size = predictive_layers
        self.out_pred_size = output_predictive_size
        
        self.in_exp_size = explanatory_input_size
        self.exp_size = explanatory_layers
        self.out_exp_size = output_explanatory_size

        self.n_filters = n_filters

        # Architecture
        self.encoder = Encoder(self.in_size, self.pred_size[0], self.pred_size[1], self.pred_size[2])
        self.decoder = Decoder(self.pred_size[2], self.pred_size[3], self.pred_size[4], self.out_pred_size)
        self.explanatory = Explanatory(self.in_exp_size, self.n_filters, self.exp_size[0], self.out_exp_size)

    def forward(self, X):

        # Predictive network
        X = self.encoder(X)
        X = self.decoder(X)
        u = X.view(X.size(0), *self.out_pred_size)

        # Manipulation of prediction output
        um = Mx(My(TensOps(u, space_dimension=2, contravariance=0, covariance=0))).values

        # Explanatory network
        K = self.explanatory(um)
        
        return u, K