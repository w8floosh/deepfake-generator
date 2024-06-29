import torch
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

class LatentSpaceDataset(Dataset):
    def __init__(self, top_latents, bottom_latents):
        self.top_latents = top_latents
        self.bottom_latents = bottom_latents
        self.length = len(top_latents)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        top_latent = torch.tensor(self.top_latents[idx]).long()
        bottom_latent = torch.tensor(self.bottom_latents[idx]).long()
        return top_latent, bottom_latent
    
class ImageDataset(Dataset):
    def __init__(self, img_dir, len, transform=None ):
        self.img_dir = img_dir
        self.transform = transform
        self.len = len
        self.img_files = [f"{str(i).zfill(5)}.jpg" for i in range(self.len)]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
