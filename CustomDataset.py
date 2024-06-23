import torch
import numpy as np
from torch.utils.data import Dataset

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
