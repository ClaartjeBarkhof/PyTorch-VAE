import torch
from torch import nn
import numpy as np

class CustomEncoder(nn.Module):
    def __init__(self, latent_dim=10, image_dim=128, hidden_dims=[32, 64, 128, 256, 512, 512]):
        super(CustomEncoder, self).__init__()

        self.image_dim = image_dim

        modules = []

        in_channels = 3
        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        # 32: 64 -> 32
        # 64: 32 -> 16
        # 128: 16 -> 8
        # 256: 8 -> 4
        # 512: 4 -> 2 (still spatially 2x2)

        # Compute the spatial resolution left after the hidden dims (divide by two every layer)
        spatial_dim_left = int((self.image_dim * (0.5 ** float(len(hidden_dims))))**2)

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Sequential(
            nn.Linear(hidden_dims[-1] * spatial_dim_left, int(hidden_dims[-1] * spatial_dim_left * 0.5)),
            nn.Linear(int(hidden_dims[-1] * spatial_dim_left * 0.5), latent_dim))
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1] * spatial_dim_left, int(hidden_dims[-1] * spatial_dim_left * 0.5)),
            nn.Linear(int(hidden_dims[-1] * spatial_dim_left * 0.5), latent_dim))

    def forward(self, image_batch):
        out = self.encoder(image_batch)
        out = out.flatten(start_dim=1)
        mu = self.fc_mu(out)
        log_var = self.fc_var(out)
        return mu, log_var


class CustomDecoder(nn.Module):
    def __init__(self, latent_dim=10, image_dim=128):
        super(CustomDecoder, self).__init__()

        self.image_dim = image_dim

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, 256 * 2)

        n_layers = int(np.log(image_dim / 4) / np.log(2))
        hidden_dims = [32 for _ in range(n_layers)]

        # hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, latent):
        print("HEey")
        out = self.decoder_input(latent)
        out = out.view(-1, 32, 4, 4)
        out = self.decoder(out)
        out = self.final_layer(out)
        return out

if __name__ == "__main__":
    image_dim = 128

    encoder = CustomEncoder(latent_dim=10, image_dim=image_dim)
    dummy_input = torch.randn((4, 3, image_dim, image_dim))
    with torch.no_grad():
        mu, log_var = encoder(dummy_input)
    print(mu.shape, log_var.shape)

    decoder = CustomDecoder(latent_dim=10, image_dim=image_dim)
    dummy_latent = torch.randn((4, 10))
    out = decoder(dummy_latent)
    print(out.shape)

