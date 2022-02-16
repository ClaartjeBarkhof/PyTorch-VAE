import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np

class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels=labels)
        val_loss = self.model.loss_function(*results,
                                            M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    #     def sample_images(self):
    #         # Get sample reconstruction image
    #         test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
    #         test_input = test_input.to(self.curr_device)
    #         test_label = test_label.to(self.curr_device)

    # #         test_input, test_label = batch
    #         recons = self.model.generate(test_input, labels = test_label)
    #         vutils.save_image(recons.data,
    #                           os.path.join(self.logger.log_dir ,
    #                                        "Reconstructions",
    #                                        f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
    #                           normalize=True,
    #                           nrow=12)

    #         try:
    #             samples = self.model.sample(144,
    #                                         self.curr_device,
    #                                         labels = test_label)
    #             vutils.save_image(samples.cpu().data,
    #                               os.path.join(self.logger.log_dir ,
    #                                            "Samples",
    #                                            f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
    #                               normalize=True,
    #                               nrow=12)
    #         except Warning:
    #             pass

    def sample_images(self):
        # print("New sample function!")
        # # Get sample reconstruction image
        # test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        # test_input = test_input.to(self.curr_device)
        # #print("test_input.shape", test_input.shape)
        # batch_size = test_input.shape[0]
        #
        # n_channels = test_input.shape[1]
        #
        # # this function is called generate but it actually reconstructs
        # recons = self.model.generate(test_input)
        # self.model.train()  # for the batch norm not to act out
        # samples = self.model.sample(batch_size, self.curr_device)
        # # print("recons.shape", recons.shape)
        # # print("samples.shape", samples.shape)
        # # print("recons min max", recons.min(), recons.max())
        # # print("samples min max", samples.min(), samples.max())
        #
        # if n_channels == 1:
        #     #print("N CHANNELS = 1")
        #     vutils.save_image(recons.data,
        #                       os.path.join(self.logger.log_dir,
        #                                    "Reconstructions",
        #                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                       normalize=False,
        #                       nrow=12)
        #
        #     vutils.save_image(samples.cpu().data,
        #                       os.path.join(self.logger.log_dir,
        #                                    "Samples",
        #                                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                       normalize=False,
        #                       nrow=12)
        # else:
        #     vutils.save_image(recons.data,
        #                       os.path.join(self.logger.log_dir,
        #                                    "Reconstructions",
        #                                    f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                       normalize=True,
        #                       value_range=(-1.0, 1.0),
        #                       nrow=12)
        #
        #
        #     vutils.save_image(samples.cpu().data,
        #                       os.path.join(self.logger.log_dir,
        #                                    "Samples",
        #                                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                       normalize=True,
        #                       value_range=(-1.0, 1.0),
        #                       nrow=12)

        print("New sample function!")

        test_input, _ = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        B, C, W, H = test_input.shape
        recons = self.model.generate(test_input).cpu() #.numpy()
        self.model.train()  # for the batch norm not to act out
        samples = self.model.sample(B, self.curr_device).cpu() #.numpy()

        # print("inital min max samples", samples.min(), samples.max())
        # print("inital min max recons", recons.min(), recons.max())

        # test_input = test_input.permute(0, 2, 3, 1)
        # test_input = (test_input + 1) * (255 / 2)
        # test_input = test_input.clamp(0, 255).to(torch.uint8).cpu().numpy()
        #
        # plt.imshow(test_input[0])
        # plt.axis("off")
        # plt.savefig(os.path.join(self.logger.log_dir, "Samples", f"TEST_INPUT_{self.logger.name}_Epoch_{self.current_epoch}.png"), dpi=200)

        ncols = 12
        nrows = int(np.ceil(B / ncols))
        subplot_wh = 0.75

        # Colour
        if C == 3:
            # Scale from -1, 1 to 0-1
            #samples = (samples + 1.0) / 2.0
            #recons = (recons + 1.0) / 2.0



            #samples = np.transpose(samples, axes=(0, 2, 3, 1))
            samples = samples.permute(0, 2, 3, 1)
            #recons = np.transpose(recons, axes=(0, 2, 3, 1))
            recons = recons.permute(0, 2, 3, 1)


            samples = (samples + 1) * (255 / 2)
            recons = (recons + 1) * (255 / 2)

            samples = samples.clamp(0, 255).to(torch.uint8).cpu().numpy()
            recons = recons.clamp(0, 255).to(torch.uint8).cpu().numpy()

            # im = im.squeeze(0).permute(1, 2, 0)
            # im = (im + 1) * (255 / 2)
            # im = im.clamp(0, 255).to(torch.uint8).cpu().numpy()

            # print("samples min max", samples.min(), samples.max())
            # print("recons min max", recons.min(), recons.max())
            #
            # print("samples.shape", samples.shape)
            # print("recons.shape", recons.shape)

        # BW
        else:
            samples = samples.numpy()
            recons = recons.numpy()

            samples = np.transpose(samples, axes=(0, 2, 3, 1)).squeeze(-1)
            recons = np.transpose(recons, axes=(0, 2, 3, 1)).squeeze(-1)

        # Plot samples
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*subplot_wh, nrows*subplot_wh))
        for im_idx in range(B):
            c = im_idx % ncols
            r = im_idx // ncols
            if C == 1:
                axs[r, c].imshow(samples[im_idx], cmap="Greys")
            else:
                axs[r, c].imshow(samples[im_idx])
            axs[r, c].axis("off")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logger.log_dir, "Samples", f"{self.logger.name}_Epoch_{self.current_epoch}.png"), dpi=75, bbox_inches = 'tight')
        #plt.show()

        # Plot reconstructions
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*subplot_wh, nrows*subplot_wh))
        for im_idx in range(B):
            c = im_idx % ncols
            r = im_idx // ncols
            if C == 1:
                axs[r, c].imshow(recons[im_idx], cmap="Greys")
            else:
                axs[r, c].imshow(recons[im_idx])
            axs[r, c].axis("off")
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.logger.log_dir, "Reconstructions", f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"), dpi=75, bbox_inches = 'tight')

        #plt.show()


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims