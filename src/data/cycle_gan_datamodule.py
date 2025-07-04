from typing import Any, Dict, Optional, Tuple, Literal, List
import os

import torch
from PIL import Image
import numpy as np
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class CycleGANDataset(Dataset):
    def __init__(
        self,
        data_dir: str = "data/summer2winter_yosemite",
        phase: Literal["train", "test"] = "train",
        img_size=256,
    ):
        super().__init__()
        if phase == "train":
            self.dir_a = os.path.join(data_dir, "trainA")
            self.dir_b = os.path.join(data_dir, "trainB")

            self.transformers = transforms.Compose(
                [
                    transforms.Resize((int(img_size * 1.5), int(img_size * 1.5))),
                    transforms.RandomCrop((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

        elif phase == "test":
            self.dir_a = os.path.join(data_dir, "trainA")
            self.dir_b = os.path.join(data_dir, "trainB")

            self.transformers = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            raise ValueError(f" not support phase:{phase}, only supprt phase and test.")

        self.image_list_a = self.__find_image(self.dir_a)
        self.image_list_b = self.__find_image(self.dir_b)

    def __find_image(self, image_dir: str) -> List[str]:
        images = os.listdir(image_dir)

        image_paths = []
        suffix = (".png", ".jpg")
        for img in images:
            if img.endswith(suffix):
                image_paths.append(os.path.join(image_dir, img))

        return image_paths

    def __len__(self):
        return min(len(self.image_list_a), len(self.image_list_b))

    def read_image(self, image_path):
        image = Image.open(image_path).convert("RGB")

        image = self.transformers(image)

        return image

    def __getitem__(self, index):
        image_a = self.read_image(self.image_list_a[index])
        image_b = self.read_image(self.image_list_b[index])

        return {"image_a": image_a, "image_b": image_b}


class CycleGANDataModule(LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 64,
        num_workers: int = 8,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)


        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = batch_size


    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        
        CycleGANDataset(self.hparams.data_dir, phase="train")
        CycleGANDataset(self.hparams.data_dir, phase="test")
        

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            
            self.data_train = CycleGANDataset(self.hparams.data_dir, phase="train")
            self.data_test = CycleGANDataset(self.hparams.data_dir, phase="test")
            self.data_val = self.data_test

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


def tensor_to_pil(x:torch.Tensor)->Image.Image:
    x = x.permute(1, 2, 0)
    x = (x * 0.5) + 0.5
    x = x * 255
    x = x.detach().cpu().numpy().astype(np.uint8)
    
    image = Image.fromarray(x)
    return image

if __name__ == "__main__":
    dataset = CycleGANDataset(phase="test")
    
    item = dataset[0]
    
    image_a = item["image_a"]
    image_b = item["image_b"]
    
    
    tensor_to_pil(image_a).save("a.png")
    
    tensor_to_pil(image_b).save("b.png")

    
