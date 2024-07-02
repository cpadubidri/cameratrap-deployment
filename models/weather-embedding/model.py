import torch
import torch.nn as nn

class FCAE(nn.Module):
    def __init__(self, input_dim, filters):
        super(FCAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, filters[0]),
            nn.ReLU(),
            nn.Linear(filters[0], filters[1]),
            nn.ReLU(),
            nn.Linear(filters[1], filters[2]),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(filters[2], filters[1]),
            nn.ReLU(),
            nn.Linear(filters[1], filters[0]),
            nn.ReLU(),
            nn.Linear(filters[0], input_dim),
            nn.Sigmoid()  # Assuming the data is normalized between 0 and 1
        )
    
    def forward(self, x):
        # Flatten the input for the encoder
        x = x.view(x.size(0), -1)
        # Encode
        latent_space = self.encoder(x)
        # Decode
        x = self.decoder(latent_space)
        # Reshape the output to the original dimensions
        x = x.view(x.size(0), 14, 12)
        return x, latent_space


if __name__=='__main__':
    input_dim = 14 * 12  # Flattened input
    filters = [512,256,256] #last number for latentspace

    # Instantiate the autoencoder
    autoencoder = FCAE(input_dim, filters)

    # Example usage
    data = torch.randn((64, 14, 12))  # Example batch size of 64
    reconstructed, latent_space = autoencoder(data)

    print("Original Data Shape:", data.shape)
    print("Reconstructed Data Shape:", reconstructed.shape)
    print("Latent Space Shape:", latent_space.shape)
