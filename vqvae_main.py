import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vqvae import VQVAE
from torchvision.utils import make_grid

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f"{str(i).zfill(5)}.jpg" for i in range(5000)]
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
def training(model, dataloader, reconstruction_criterion, optimizer,epoch):
    model.train()
    running_loss = 0.0
    for inputs in dataloader:
        #inputs = inputs.to(device)
        optimizer.zero_grad()
        outputs, vq_loss = model(inputs)
        reconstruction_loss = reconstruction_criterion(outputs, inputs)
        commitment_loss = 0.3 * vq_loss.mean()
        loss = reconstruction_loss + commitment_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        print(running_loss)

    avg_loss = running_loss / len(dataloader)
    torch.save(model.state_dict(), f'vqvae_epoch_{epoch+1}.pt')
    return avg_loss

def validate(model, dataloader, reconstruction_criterion):
    model.eval()  
    total_loss = 0.0
    with torch.no_grad():
        for inputs in dataloader:
            outputs, vq_loss = model(inputs)
            reconstruction_loss = reconstruction_criterion(outputs, inputs)
            commitment_loss = 0.25 * vq_loss.mean()
            loss = reconstruction_loss + commitment_loss
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss
    
def show_images(images, nrow=1):
    grid_img = make_grid(images, nrow=nrow, normalize=True) 
    # Convert the tensor to a numpy array
    np_img = grid_img.permute(1, 2, 0).cpu().numpy()
    # Display the images
    plt.figure(figsize=(15, 8))
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()

def main():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
    img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'
    dataset = ImageDataset(img_dir=img_dir, transform=transform)
    train_size = 3500  # Number of samples for training
    test_size = len(dataset) - train_size  # Remaining samples for testing
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    pretrained_weights = torch.load('C:/Users/sabry/Downloads/vqvae_560.pt')
    model = VQVAE()
    model.load_state_dict(pretrained_weights)

    # Loss functions
    reconstruction_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Fine-tuning loop
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss = training(model, train_dataloader, reconstruction_criterion,optimizer,epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}")
        val_loss = validate(model, test_dataloader, reconstruction_criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")
if __name__ == "__main__":
    main()
    