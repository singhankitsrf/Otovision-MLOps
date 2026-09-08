from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .preprocessing import crop_dark_border


class OtoscopyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: Callable, label_to_index=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform
        labels = sorted(self.frame["label"].unique().tolist())
        self.label_to_index = label_to_index or {label: i for i, label in enumerate(labels)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(Path(row["path"])).convert("RGB")
        image = crop_dark_border(image)
        image = self.transform(image)
        label = self.label_to_index[row["label"]]
        return image, label


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.03),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, eval_transform
