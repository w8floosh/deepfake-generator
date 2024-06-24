import os
import argparse
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
from pixelsnail import PixelSNAIL
from CustomDataset import LatentSpaceDataset
from tqdm import tqdm

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

def train_pixelsnail(train_loader, model, type, optimizer):

    loader = tqdm(train_loader)
    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom) in enumerate(loader):
        model.zero_grad()
        
        if type == 'top':
            #top = top.squeeze()
            target = top
            out, _ = model(top)

        else:
            #bottom = bottom.squeeze()
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()
        
    return loss, accuracy

def main():

    parser = argparse.ArgumentParser(description='Train or test the models')
    parser.add_argument('--mode', choices=['train', 'test', 'generate'], help='Specify whether to train or test the models or generate model results')
    parser.add_argument('--model', choices=['vqvae', 'pixelsnail'],help='Specify the model')
    #parser.add_argument('--label', help='Path to the label file (required for the predict mode)')
    args = parser.parse_args()

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
    img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'

    dataset = ImageDataset(img_dir=img_dir, len=10000, transform=transform)
    train_size = int(dataset.len * 0.7) # Number of samples for training
    test_size = dataset.len - train_size  # Remaining samples for testing
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    test_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + test_size))
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=2)
    
    #pretrained_weights = torch.load('C:/Users/sabry/Downloads/vqvae_560.pt')
    pretrained_weights = torch.load('./vqvae_epoch_10.pt', map_location=torch.device('cpu'))
    model = VQVAE()
    model.load_state_dict(pretrained_weights)

    if args.mode == "train" and args.model == "vqvae":
        reconstruction_criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 0
        
        for epoch in range(num_epochs):
            train_loss = training(model, train_dataloader, reconstruction_criterion,optimizer,epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}")
            val_loss = validate(model, test_dataloader, reconstruction_criterion)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")
            
            with open('losses.txt', 'a') as f:
                f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Validation Loss: {val_loss}\n")
    elif args.mode =="train" and args.model =="pixelsnail":
    
        data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        top_latents = []
        bottom_latents = []

        models ={ "top": PixelSNAIL([32, 32],512,256,5,4,4,128,dropout=0.1,n_out_res_block=2), 
                 "bottom": PixelSNAIL([64, 64],512,256,5,4,4,128,attention=False,dropout=0.1,n_cond_res_block=3, cond_res_channel=256)
                }
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        for space in ["top","bottom"]:
            for epoch in range(num_epochs):
                total_loss = 0.0
                total_accuracy = 0.0
                for i, image in enumerate(data_loader):
                    with torch.no_grad():
                        _, _, _, top_latents, bottom_latents = model.encode(image)

                    top_latents = np.array(top_latents)
                    bottom_latents = np.array(bottom_latents)
                    
                    latent_space_dataset = LatentSpaceDataset(top_latents, bottom_latents)
                    latent_space_loader = DataLoader(latent_space_dataset, batch_size=64, shuffle=True)

                    pixelsnail_loss, accuracy = train_pixelsnail(latent_space_loader, models[space], space, optimizer)
                    total_loss+= pixelsnail_loss
                    total_accuracy+= accuracy
                    
                num_batch = dataset.len / batch_size
                print(f'{space} - Epoch [{epoch+1}/{num_epochs}], Loss: {pixelsnail_loss / num_batch:.4f}, Accuracy: {accuracy / num_batch}')
                with open('pixelsnail_losses.txt', 'a') as f:
                    f.write(f"{space} - Epoch [{epoch+1}/{num_epochs}], Loss: {pixelsnail_loss}, Accuracy: {accuracy}\n")

def imshow(img, title=None):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title)
    plt.show()
    
def show_side_by_side(original, reconstructed):
    original = np.transpose(original.squeeze().numpy(), (1, 2, 0)) 
    reconstructed = np.transpose(reconstructed.squeeze().numpy(), (1, 2, 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].axis('off')  
    axes[0].set_title('original') 

    axes[1].imshow(reconstructed)
    axes[1].axis('off') 
    axes[1].set_title('reconstructed') 

    plt.tight_layout()
    plt.show()
    
def evaluate():

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    model = VQVAE()
    model.load_state_dict(torch.load('vqvae_epoch_10.pt', map_location=torch.device('cpu')))  # Load your trained model
    model.eval()
    img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'
    dataset = ImageDataset(img_dir=img_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)
    
    dataiter = iter(dataloader)
    images, _ = next(dataiter)
    print(f"Input images shape: {images.shape}")
    if len(images.shape) == 3:
        images = images.unsqueeze(0)
    
    with torch.no_grad():
        reconstructed, _ = model(images)
    images = torch.clamp(images, 0, 1)
    print(images.shape)
    reconstructed = torch.clamp(reconstructed, 0, 1)
    show_side_by_side(images, reconstructed)

if __name__ == "__main__":
    main()
    #evaluate()