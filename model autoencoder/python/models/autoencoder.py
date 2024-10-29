import torch
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, input_size, latent_space_dim, device):
        super(Encoder, self).__init__()

        # Encoder parameters
        self.input = input_size
        self.latent_space = latent_space_dim
        self.device = device

        # Encoder structure
        self.flatten_layer = nn.Flatten(start_dim=1, end_dim=-1)
        self.hidden_1_layer = nn.Linear(torch.prod(torch.tensor(self.input, device=self.device)), 
                                        torch.prod(torch.tensor(self.input, device=self.device))//2).to(self.device)
        self.hidden_2_layer = nn.Linear(torch.prod(torch.tensor(self.input, device=self.device))//2, 
                                        self.latent_space).to(self.device)
        
    def forward(self, X):
        X = self.flatten_layer(X)
        X = torch.sigmoid(self.hidden_1_layer(X))
        encoder_output = torch.sigmoid(self.hidden_2_layer(X))

        return encoder_output


class Decoder(nn.Module):

    def __init__(self, latent_space_dim, output_size, device):
        super(Decoder, self).__init__()

        # Decoder parameters
        self.latent_space = latent_space_dim
        self.output = output_size
        self.device = device

        # Decoder structure
        self.hidden_1_layer = nn.Linear(self.latent_space, torch.prod(torch.tensor(self.output, device=self.device))//2).to(self.device)
        self.hidden_2_layer = nn.Linear(torch.prod(torch.tensor(self.output, device=self.device))//2, 
                                        torch.prod(torch.tensor(self.output, device=self.device))).to(self.device)
        
    def forward(self, X):
        
        X = torch.sigmoid(self.hidden_1_layer(X))
        X = self.hidden_2_layer(X)
        decoder_output = X.view(X.size(0), self.output[0], self.output[1], self.output[2])
        return decoder_output


class Autoencoder(nn.Module):

    def __init__(self, input_size, encoding_dim, output_size, device):
        super(Autoencoder, self).__init__()
        self.device = device
        self.encoder = Encoder(input_size, encoding_dim, device).to(device)
        self.decoder = Decoder(encoding_dim, output_size, device).to(device)
        
    def forward(self, x):
        x = self.encoder(x.to(self.device))
        x = self.decoder(x)
        return x