import torch
# from models import BaseVAE
from torch import nn
from torch.nn import functional as F
import math

# Adapted DCGAN Generator (adapted as decoder) and Discriminator (adapted as encoder)
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class DCGAN_Decoder(nn.Module):
    def __init__(self, latent_dim=10, n_feature_maps=64, num_channels=3):
        super(DCGAN_Decoder, self).__init__()

        self.ngf = n_feature_maps  # number of generator feature maps
        self.nz = latent_dim
        self.nc = num_channels

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.nz, self.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # batch, latent_dim -> batch, latent_dim, 1, 1
        input = input[:, :, None, None]
        # input = input.unsqueeze(-1).unsqueeze(-1)
        # print(input.shape)
        return self.main(input)


class DCGAN_Encoder(nn.Module):
    def __init__(self, latent_dim=10, n_feature_maps=64, num_channels=3):
        super(DCGAN_Encoder, self).__init__()

        self.ndf = n_feature_maps  # number of generator feature maps
        self.nz = latent_dim
        self.nc = num_channels

        self.conv_net = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(self.ndf * 8, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.pre_posterior_layer = torch.nn.Linear(1024, 256)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)

    def forward(self, input):
        out = self.conv_net(input)
        out = out.flatten(start_dim=1)
        out = self.pre_posterior_layer(out)

        mu = self.fc_mu(out)
        logvar = self.fc_mu(out)

        return mu, logvar