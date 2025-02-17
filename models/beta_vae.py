import torch
from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from .types_ import *
import math
from .DCGAN_encoder_decoder import DCGAN_Encoder, DCGAN_Decoder
from .custom_architectures import CustomEncoder, CustomDecoder


class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 image_dim: int = 64,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 dcgan: bool = False,
                 custom_architecture: bool = False,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()

        assert not (dcgan and custom_architecture), "either dcgan or custom_architecture may be true, not both"

        print(f"Using DCGAN architecture = {dcgan}")
        print(f"Using custom architecture = {custom_architecture}")
        print(f"In channels:", in_channels)

        self.channels = in_channels
        self.latent_dim = latent_dim
        self.image_dim = image_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        self.dcgan = dcgan
        self.custom_architecture = custom_architecture

        if not (self.dcgan or self.custom_architecture):
            modules = []
            if hidden_dims is None:
              hidden_dims = [32, 64, 128, 256, 512]

            # Build Encoder
            for h_dim in hidden_dims:
              modules.append(
                  nn.Sequential(
                      nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size= 3, stride= 2, padding  = 1),
                      nn.BatchNorm2d(h_dim),
                      nn.LeakyReLU())
              )
              in_channels = h_dim

            self.encoder = nn.Sequential(*modules)
            self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
            self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


            # Build Decoder
            modules = []

            self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

            hidden_dims.reverse()

            for i in range(len(hidden_dims) - 1):
              modules.append(
                  nn.Sequential(
                      nn.ConvTranspose2d(hidden_dims[i],
                                        hidden_dims[i + 1],
                                        kernel_size=3,
                                        stride = 2,
                                        padding=1,
                                        output_padding=1),
                      nn.BatchNorm2d(hidden_dims[i + 1]),
                      nn.LeakyReLU())
              )

            self.decoder = nn.Sequential(*modules)

            if self.channels == 3:
                print("IN CHANNELS = 3")
                self.final_layer = nn.Sequential(
                                  nn.ConvTranspose2d(hidden_dims[-1],
                                                    hidden_dims[-1],
                                                    kernel_size=3,
                                                    stride=2,
                                                    padding=1,
                                                    output_padding=1),
                                  nn.BatchNorm2d(hidden_dims[-1]),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(hidden_dims[-1], out_channels= self.channels,
                                            kernel_size= 3, padding= 1),
                                  nn.Tanh())
            else:
                print("IN CHANNELS = 1")
                self.final_layer = nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[-1],
                                       hidden_dims[-1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[-1]),
                    nn.LeakyReLU(),
                    nn.Conv2d(hidden_dims[-1], out_channels=self.channels,
                              kernel_size=3, padding=1),
                    nn.Sigmoid())
        else:
            if self.dcgan:
                self.encoder = DCGAN_Encoder(latent_dim=latent_dim, num_channels=in_channels, n_feature_maps=64)
                self.decoder = DCGAN_Decoder(latent_dim=latent_dim, num_channels=in_channels, n_feature_maps=64)
            else:
                self.encoder = CustomEncoder(latent_dim=latent_dim, image_dim=self.image_dim, in_channels=in_channels)
                self.decoder = CustomDecoder(latent_dim=latent_dim, image_dim=self.image_dim, channels=in_channels)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        # print("INPUT SHAPE", input.shape)

        if not (self.dcgan or self.custom_architecture):
          result = self.encoder(input)
          result = torch.flatten(result, start_dim=1)

          # Split the result into mu and var components
          # of the latent Gaussian distribution
          mu = self.fc_mu(result)
          log_var = self.fc_var(result)
        else:
          # print("beta vae encoder")
          mu, log_var = self.encoder(input)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:

        if not (self.dcgan or self.custom_architecture):
          result = self.decoder_input(z)
          #print("1", result.shape)
          result = result.view(-1, 512, 2, 2)
          #print("2", result.shape)
          result = self.decoder(result)
          #print("3", result.shape)
          result = self.final_layer(result)
          #print("4", result.shape)
        else:
          # print("beta vae decoder")
          result = self.decoder(z)

        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        weighted_kl = 0.0

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            weighted_kl = self.beta * kld_weight * kld_loss
            loss = recons_loss + weighted_kl
            return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss, "weighted_kl": weighted_kl}  # , 'weighted_kl':weighted_kl
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss} #, 'weighted_kl':weighted_kl

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

if __name__ == "__main__":
    vae = BetaVAE(in_channels=3, latent_dim=10, image_dim=64, dcgan=False, custom_architecture=True)
    dummy_input = torch.randn((4, 3, 64, 64))
    with torch.no_grad():
        out = vae(dummy_input)