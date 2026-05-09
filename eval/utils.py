import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class RGBFIDDataset(Dataset):
    def __init__(self, root_dir, crop_settings=None, image_size=256, aug_idx = 1):
        self.root_dir = root_dir
        self.aug_idx = aug_idx
        self.files = sorted(os.listdir(root_dir))
        self.crop_settings = crop_settings
        self.image_size = image_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root_dir, self.files[idx])
        img = Image.open(path).convert("RGB")

        if self.crop_settings is not None:
            x1, y1, x2, y2, flip = self.crop_settings[self.aug_idx]
            top, left = y1, x1
            h, w = (y2 - y1), (x2 - x1)
            img = TF.crop(img, top, left, h, w)
            if flip:
                img = TF.hflip(img)

        img = img.resize((self.image_size, self.image_size))
        img = np.array(img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)

        return img