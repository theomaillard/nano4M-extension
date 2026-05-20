import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

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
        img = torch.from_numpy(img).permute(2, 0, 1).byte()
    
        return img

def compute_fid(
    real_dir,
    fake_dir,
    crop_settings_dir=None,
    batch_size=32,
    device="cuda",
    image_size=256,
    num_workers=4,
    max_images=None
):
    crop_settings = np.load(crop_settings_dir) if crop_settings_dir else None
    
    real_dataset = RGBFIDDataset(
        root_dir=real_dir,
        crop_settings=crop_settings,
        image_size=image_size,
    )

    fake_dataset = RGBFIDDataset(
        root_dir=fake_dir,
        crop_settings=crop_settings,
        image_size=image_size,
    )

    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    fake_loader = DataLoader(
        fake_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    fid = FrechetInceptionDistance(
        feature=2048,
        normalize=True,
    ).to(device)

    fid.reset()

    num_real = 0
    
    for imgs in tqdm(real_loader, desc="Real images"):

        if max_images is not None:

            remaining = max_images - num_real

            if remaining <= 0:
                break
                
            imgs = imgs[:remaining].contiguous()
            
        imgs = imgs.to(device)
        fid.update(imgs, real=True)

        num_real += imgs.size(0)


    num_fake = 0

    for imgs in tqdm(fake_loader, desc="Fake images"):
        
        if max_images is not None:

            remaining = max_images - num_fake

            if remaining <= 0:
                break
                
            imgs = imgs[:remaining].contiguous()
            
        imgs = imgs.to(device)
        fid.update(imgs, real=False)

        num_fake += imgs.size(0)

    score = fid.compute()

    return score.item()

def preprocess_single_image(path, crop_settings=None, aug_idx=1, image_size=256):
    img = Image.open(path).convert("RGB")

    if crop_settings is not None:
        x1, y1, x2, y2, flip = crop_settings[aug_idx]
        img = TF.crop(img, y1, x1, y2 - y1, x2 - x1)

        if flip:
            img = TF.hflip(img)

    img = img.resize((image_size, image_size))
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

    return img.unsqueeze(0)
    

def compute_ssim(
    img1,
    img2,
    device="cuda",
    data_range=1.0
):

    img1 = img1.to(device)
    img2 = img2.to(device)

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    score = ssim_metric(img1, img2)

    return score.item()

    
    