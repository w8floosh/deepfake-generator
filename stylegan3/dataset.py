import os
from torch.utils.data import Dataset
from PIL import Image


# Custom Dataset Class for CelebA-HQ
class CelebAHQDataset(Dataset):
    def __init__(self, img_dir, transform=None, num_images=10000, index=0):
        self.img_dir = img_dir
        self.transform = transform
        self.image_files = os.listdir(img_dir)[index:num_images]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
