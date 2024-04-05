import gc

from torch import Tensor
import numpy as np
import pickle
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torch.utils.data.distributed import DistributedSampler


def load_data(
    data_dir,
    input_channels,
    image_size,
    mode: list,
    batch_size: int,
    n_workers: int,
    class_cond: bool = True,
    transforms=None,
) -> tuple[DataLoader, DistributedSampler]:
    data_train = CustomDataset(
        data_dir=data_dir,
        input_channels=input_channels,
        image_size=image_size,
        mode=mode,
        class_cond=class_cond,
        transforms=transforms,
    )
    sampler = DistributedSampler(data_train)
    loader = DataLoader(
        data_train,
        batch_size=batch_size,
        num_workers=n_workers,
        sampler=sampler,
        drop_last=True,
    )
    return loader, sampler


def transback(data: Tensor) -> Tensor:
    return data / 2 + 0.5


class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir,
        input_channels=3,
        image_size=32,
        mode=["train"],
        class_cond=True,
        transforms=None,
    ):
        with open(data_dir, "rb") as f:
            data = pickle.load(f)
        self.classes = (
            None
            if not class_cond
            else np.empty(
                [
                    0,
                ]
            )
        )
        self.labels = (
            None
            if not class_cond
            else np.empty(
                [
                    0,
                ]
            )
        )
        self.images = np.empty([0, input_channels, image_size, image_size])
        self.class_cond = class_cond
        self.transforms = transforms
        for m in mode:
            self.images = np.concatenate(
                [self.images, data[f"{m}_clean"]["image"], data[f"{m}_noisy"]["image"]]
            )
            if class_cond:
                self.labels = np.concatenate(
                    [
                        self.labels,
                        data[f"{m}_clean"]["label"],
                        data[f"{m}_noisy"]["label"],
                    ]
                )
                self.classes = np.concatenate(
                    [
                        self.classes,
                        data[f"{m}_clean"]["class"],
                        data[f"{m}_noisy"]["class"],
                    ]
                )
        del data
        gc.collect()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        kwargs = {}
        if self.class_cond:
            kwargs["y"] = self.labels[item]
            kwargs["y_true"] = self.classes[item]

        return (
            self.transforms(self.images[item]) if self.transforms else self.images[item]
        )
