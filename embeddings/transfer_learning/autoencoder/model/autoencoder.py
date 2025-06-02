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
    
class Autoencoder(nn.Module):
    
    def __init__(self, input_size, predictive_layers, output_size):
        
        super(Autoencoder, self).__init__()

        # Parameters
        self.in_size = input_size
        self.pred_size = predictive_layers
        self.out_size = output_size
        
        # Architecture
        self.encoder = Encoder(self.in_size, self.pred_size[0], self.pred_size[1], self.pred_size[2])
        self.decoder = Decoder(self.pred_size[2], self.pred_size[3], self.pred_size[4], self.out_size)

    def forward(self, X):

        # Predictive network
        X = self.encoder(X)
        X = self.decoder(X)
        u = X.view(X.size(0), *self.out_size)

        return u