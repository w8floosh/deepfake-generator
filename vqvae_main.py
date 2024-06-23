import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vqvae import VQVAE
from torchvision.utils import make_grid

class ImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_files = [f"{str(i).zfill(5)}.jpg" for i in range(10000)]
        
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
    latent_loss_weight = 0.25
    for inputs in dataloader:
        #inputs = inputs.to(device)
        model.zero_grad()
        outputs, latent_loss = model(inputs)
        reconstruction_loss = reconstruction_criterion(outputs, inputs)
        #commitment_loss = 0.3 * vq_loss.mean()
        #loss = reconstruction_loss + commitment_loss
        latent_loss = latent_loss.mean()
        loss = reconstruction_loss + latent_loss_weight * latent_loss
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
    latent_loss_weight = 0.25
    with torch.no_grad():
        for inputs in dataloader:
            outputs, latent_loss = model(inputs)
            reconstruction_loss = reconstruction_criterion(outputs, inputs)
            latent_loss = latent_loss.mean()
            loss =  reconstruction_loss + latent_loss_weight * latent_loss
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
    transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    #transforms.CenterCrop(args.size)
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
    #img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'
    img_dir = 'C:/Users/markn/Downloads/Dataset/celeba_hq_256'
    dataset = ImageDataset(img_dir=img_dir, transform=transform)
    train_size = 7000  # Number of samples for training
    test_size = len(dataset) - train_size  # Remaining samples for testing
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    #pretrained_weights = torch.load('C:/Users/sabry/Downloads/vqvae_560.pt')
    pretrained_weights = torch.load('./vqvae_560.pt', map_location=torch.device('cpu'))
    model = VQVAE()
    model.load_state_dict(pretrained_weights)

    # Loss functions
    reconstruction_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Fine-tuning loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        train_loss = training(model, train_dataloader, reconstruction_criterion,optimizer,epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}")
        val_loss = validate(model, test_dataloader, reconstruction_criterion)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")
        
        with open('losses.txt', 'a') as f:
            f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Validation Loss: {val_loss}\n")

# Function to visualize images
def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()
    
def show_side_by_side(original, reconstructed):
    original = np.transpose(original.squeeze().numpy(), (1, 2, 0)) 
    reconstructed = np.transpose(reconstructed.squeeze().numpy(), (1, 2, 0))
    # Create a figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Display image 1
    axes[0].imshow(original)
    axes[0].axis('off')  # Optional: turn off axis ticks and labels
    axes[0].set_title('original')  # Optional: set a title

    # Display image 2
    axes[1].imshow(reconstructed)
    axes[1].axis('off')  # Optional: turn off axis ticks and labels
    axes[1].set_title('reconstructed')  # Optional: set a title

    plt.tight_layout()
    plt.show()
    
def evaluate():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
    ])
    model = VQVAE()
    model.load_state_dict(torch.load('vqvae_epoch_10.pt', map_location=torch.device('cpu')))  # Load your trained model
    model.eval()
    img_dir = 'C:/Users/markn/Downloads/Dataset/celeba_hq_256'
    dataset = ImageDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
    
    dataiter = iter(dataloader)
    images, _ = next(dataiter)
    print(f"Input images shape: {images.shape}")
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    images = torch.clamp(images, 0, 1)
    print(images.shape)
    #print("Original Images")
    #imshow(torchvision.utils.make_grid(images))
    reconstructed = torch.clamp(reconstructed, 0, 1)
    #print("Reconstructed Images")
    #imshow(torchvision.utils.make_grid(reconstructed))
    show_side_by_side(images, reconstructed)
    
def generator():
    model = VQVAE()
    model.load_state_dict(torch.load('vqvae_epoch_10.pt', map_location=torch.device('cpu')))  # Load your trained model
    model.eval()

    # Set the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model.to(device)

    # Generate new samples
    num_samples = 10  # Number of samples to generate
    samples = model.generate(num_samples, device)

    # Move the samples to CPU and convert to numpy
    samples = samples.detach().cpu().numpy()

    # Ensure the samples are in the valid range
    samples = np.clip(samples, 0, 1)  # Assuming samples are floats in the range [0, 1]

    # Plot generated samples
    for i in range(num_samples):
        plt.imshow(samples[i].transpose(1, 2, 0))  # Adjust dimensions if necessary
        plt.show()

if __name__ == "__main__":
    #main()
    #evaluate()
    generator()