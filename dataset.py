import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile


# Add your custom dataset class here
class SyntheticDataset(Dataset):
    def __init__(self,
                 data_dir: str,
                 split: str,
                 transform: Callable,
                 **kwargs):

        self.data_dir = Path(data_dir)
        print("datadir=", self.data_dir)
        self.transforms = transform

        if "celeb" in str(self.data_dir):
            imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        else:
            imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.png'])

        self.imgs = imgs[:int(len(imgs) * 0.9)] if split == "train" else imgs[int(len(imgs) * 0.9):]
        print(split, "len(imgs)", len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """

    def __init__(self,
                 data_path: str,
                 split: str,
                 transform: Callable,
                 **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])

        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])

        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy datat to prevent breaking


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            image_dir: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = image_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        print(f"Setting up {self.patch_size}: {self.data_dir}")

        if  "celeba" in self.data_dir:
            print("Using CelebA")
            print("*"* 40)
            print("** Warning: missing normalize")
            data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.CenterCrop(148),
                                                  transforms.Resize(self.patch_size),
                                                  transforms.ToTensor(),])

        elif "gradient" in self.data_dir or "random" in self.data_dir or "synthetic_6" in self.data_dir:
            print("Using synthetic coloured dataset")
            data_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.Resize(self.patch_size),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5330, 0.5264, 0.5296),
                                                                        (0.3773, 0.3776, 0.3732))])

        elif "bw" in self.data_dir:
            print("Using synthetic BW dataset")
            bw_mean = 0.0441
            bw_std = 0.5247
            data_transforms = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.Resize(self.patch_size),
                                                  transforms.ToTensor()])
            # ,
            #                                                   transforms.Normalize(bw_mean, bw_std)

        else:
            raise NotImplementedError

        self.train_dataset = SyntheticDataset(
            self.data_dir,
            split='train',
            transform=data_transforms,
        )

        self.val_dataset = SyntheticDataset(
            self.data_dir,
            split='val',
            transform=data_transforms,
        )

    #       ===============================================================

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
