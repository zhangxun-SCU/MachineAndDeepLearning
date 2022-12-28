import torch
from PIL import Image
import os
import config
from torch.utils.data import Dataset
import numpy as np


class MyDataSet(Dataset):
    def __init__(self, root_a, root_b, transform=None):
        super(MyDataSet, self).__init__()
        self.root_a = root_a
        self.root_b = root_b
        self.transform = transform

        self.a_images = os.listdir(root_a)
        self.b_images = os.listdir(root_b)

        self.length_dataset = max((len(self.a_images), len(self.b_images)))

        self.a_len = len(self.a_images)
        self.b_len = len(self.b_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, item):
        a_img = self.a_images[item % self.a_len]
        b_img = self.b_images[item % self.b_len]

        a_path = os.path.join(self.root_a, a_img)
        b_path = os.path.join(self.root_b, b_img)

        a_img = np.array(Image.open(a_path).convert("RGB"))
        b_img = np.array(Image.open(b_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=a_img, image0=b_img)
            a_img = augmentations["image"]
            b_img = augmentations["image0"]

        return a_img, b_img
