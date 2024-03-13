"""Custom faces dataset."""
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class FacesDataset(Dataset):
    """Faces dataset.

    Attributes:
        root_path: str. Directory path to the dataset. This path has to
        contain a subdirectory of real images called 'real' and a subdirectory
        of not-real images (fake / synthetic images) called 'fake'.
        transform: torch.Transform. Transform or a bunch of transformed to be
        applied on every image.
    """
    def __init__(self, root_path: str, transform=None):
        """Initialize a faces dataset."""
        self.root_path = root_path
        self.real_image_names = os.listdir(os.path.join(self.root_path, 'real'))
        self.fake_image_names = os.listdir(os.path.join(self.root_path, 'fake'))
        self.transform = transform

    def __getitem__(self, index) -> tuple[torch.Tensor, int]:
        """Get a sample and label from the dataset."""
        """INSERT YOUR CODE HERE, overrun return."""
        with torch.no_grad():
            if index < len(self.real_image_names):
                label = 0  # if index is smaller than real images len, choose real image
                image = Image.open(os.path.join(self.root_path, 'real', self.real_image_names[index]))
            else:
                label = 1  # if index is equal/larger than real images len, choose fake image
                image = Image.open(os.path.join(self.root_path, 'fake', self.fake_image_names[index-len(self.real_image_names)]))

            if self.transform is not None:
                image = self.transform(image)

            sample = (image, label)

        return sample

    def __len__(self):
        """Return the number of images in the dataset."""
        return len(self.fake_image_names)+len(self.real_image_names)
