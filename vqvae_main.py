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
from CustomDataset import *
from tqdm import tqdm
from pixelcnn import PixelCNN
from utils import *
from torch.autograd import Variable

def train_vqvae(model, dataloader, reconstruction_criterion, optimizer, epoch, device):
    model.train()
    running_loss = 0.0
    latent_loss_weight = 0.25
    for inputs in dataloader:
        inputs = inputs.to(device)
        model.zero_grad()
        outputs, latent_loss = model(inputs)
        reconstruction_loss = reconstruction_criterion(outputs, inputs)
        latent_loss = latent_loss.mean()
        loss = reconstruction_loss + latent_loss_weight * latent_loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        print(loss.item())

    avg_loss = running_loss / len(dataloader.dataset)
    torch.save(model.state_dict(), f'vqvae_epoch_{epoch+1}.pt')
    return avg_loss

def val_vqvae(model, dataloader, reconstruction_criterion, device):
    model.eval()
    total_loss = 0.0
    latent_loss_weight = 0.25
    with torch.no_grad():
        for inputs in dataloader:
            inputs = inputs.to(device)
            outputs, latent_loss = model(inputs)
            reconstruction_loss = reconstruction_criterion(outputs, inputs)
            latent_loss = latent_loss.mean()
            loss =  reconstruction_loss + latent_loss_weight * latent_loss
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def show_images(images, nrow=1):
    grid_img = make_grid(images, nrow=nrow, normalize=True)
    np_img = grid_img.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(15, 8))
    plt.imshow(np_img)
    plt.axis('off')
    plt.show()

# def train_pixelsnail(train_loader, model, type, optimizer):

#     loader = tqdm(train_loader)
#     criterion = nn.CrossEntropyLoss()

#     for i, (top, bottom) in enumerate(loader):
#         model.zero_grad()

#         if type == 'top':
#             #top = top.squeeze()
#             target = top
#             out, _ = model(top)

#         else:
#             #bottom = bottom.squeeze()
#             target = bottom
#             out, _ = model(bottom, condition=top)

#         loss = criterion(out, target)
#         loss.backward()

#         optimizer.step()

#         _, pred = out.max(1)
#         correct = (pred == target).float()
#         accuracy = correct.sum() / target.numel()

#     return loss, accuracy

def normalize(tensor):
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_input = (tensor - min_value) / (max_value - min_value)
    scaled_input = 2.0 * normalized_input - 1.0
    return scaled_input.unsqueeze(1)

def train_pixelcnn(train_loader, vqvae, model, space, optimizer, loss_op, epoch, device):
    model.train()
    train_loss = 0.
    for i, image in enumerate(train_loader):
        image = image.to(device)
        print("image size",image.size(0))
        with torch.no_grad():
            _, _, _, top_latents, bottom_latents = vqvae.encode(image)
        if space == "top":
            input = top_latents
        else:
            input = bottom_latents
        
        input = normalize(input)
        input = Variable(input)
        output = model(input)
    
        loss = loss_op(input, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * image.size(0)
        print(loss.item())

    avg_loss = train_loss / len(train_loader.dataset)
    torch.save(model.state_dict(), f'pixelcnn_epoch_{epoch+1}.pt')
    return avg_loss

def val_pixelcnn(val_loader, vqvae, model, space, loss_op, device):
    model.eval()
    test_loss = 0.
    for i, image in enumerate(val_loader):
        image = image.to(device)
        with torch.no_grad():
            _, _, _, top_latents, bottom_latents = vqvae.encode(image)
        if space == "top":
            input = top_latents
        else:
            input = bottom_latents

        input = normalize(input)
        input_var = Variable(input)

        output = model(input_var)
        loss = loss_op(input_var, output)
        test_loss += loss.item() * image.size(0)

    avg_loss = test_loss / len(val_loader.dataset)
    return avg_loss
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parser = argparse.ArgumentParser(description='Train or test the models')
    parser.add_argument('--mode', choices=['train', 'test', 'generate'], help='Specify whether to train or test the models or generate model results')
    parser.add_argument('--model', choices=['vqvae', 'pixelcnn'],help='Specify the model')
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'

    dataset = ImageDataset(img_dir=img_dir, len=10000, transform=transform)
    train_size = int(dataset.len * 0.7) 
    val_size = dataset.len - train_size  
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=2)

    #pretrained_weights = torch.load('C:/Users/sabry/Downloads/vqvae_560.pt')
    pretrained_weights = torch.load('./vqvae_epoch_10.pt', map_location=torch.device('cuda'))
    vqvae = VQVAE().to(device)
    vqvae.load_state_dict(pretrained_weights)

    if args.mode == "train" and args.model == "vqvae":
        reconstruction_criterion = nn.MSELoss()
        optimizer = optim.Adam(vqvae.parameters(), lr=0.0001)

        num_epochs = 10

        for epoch in range(num_epochs):
            train_loss = train_vqvae(vqvae, train_dataloader, reconstruction_criterion,optimizer,epoch, device)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}")
            val_loss = val_vqvae(vqvae, val_dataloader, reconstruction_criterion, device)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")

            with open('losses.txt', 'a') as f:
                f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Validation Loss: {val_loss}\n")

    elif args.mode =="train" and args.model == "pixelcnn":
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        nr_logistic_mix = 20
        top_model = PixelCNN(nr_resnet=5, nr_filters= 160, input_channels=1, nr_logistic_mix=nr_logistic_mix).to(device)
        bottom_model = PixelCNN(nr_resnet=5, nr_filters= 160, input_channels=1, nr_logistic_mix=nr_logistic_mix).to(device)
        
        path = "C:/Users/sabry/Downloads/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth"

        load_part_of_model(top_model, path)
        load_part_of_model(bottom_model, path)
        
        models = { 
            "top": top_model,
            "bottom": bottom_model
        }
        
        loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
        num_epochs = 5
        for space in ["top","bottom"]:
            optimizer = optim.Adam(models[space].parameters(), lr=0.0004)
            for epoch in range(num_epochs):
                train_loss = train_pixelcnn(train_loader, vqvae, models[space],space,optimizer,loss_op, epoch, device)
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}")
                val_loss = val_pixelcnn(val_loader, vqvae, models[space], space, loss_op, device)
                print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")

                with open('pixelcnn_losses.txt', 'a') as f:
                    f.write(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}, Validation Loss: {val_loss}\n")

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

def evaluate_vqvae():

    transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    model = VQVAE()
    model.load_state_dict(torch.load('vqvae_epoch_15.pt', map_location=torch.device('cpu')))  # Load your trained model
    model.eval()
    img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'
    dataset = ImageDataset(img_dir=img_dir, len=2, transform=transform)
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

def transform_range(output_tensor, min_original, max_original):
    min_output = torch.min(output_tensor)
    max_output = torch.max(output_tensor)
    transformed_tensor = (output_tensor - min_output) * (max_original - min_original) / (max_output - min_output) + min_original

    return transformed_tensor

def prova(vqvae):
    nr_logistic_mix = 20
    top_model = PixelCNN(nr_resnet=5, nr_filters= 160, input_channels=1, nr_logistic_mix=nr_logistic_mix)
    bottom_model = PixelCNN(nr_resnet=5, nr_filters= 160, input_channels=1, nr_logistic_mix=nr_logistic_mix)
    
    path = "C:/Users/sabry/Downloads/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth"

    load_part_of_model(top_model, path)
    load_part_of_model(bottom_model, path)

    top_model.eval()
    bottom_model.eval()

    # top_model = top_model.cuda()
    # bottom_model = bottom_model.cuda()
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    img_dir = 'C:/Users/sabry/Downloads/Dataset/celeba_hq_256'
    dataset = ImageDataset(img_dir=img_dir, len=2, transform=transform)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for i, images in enumerate(data_loader):
        with torch.no_grad():
            _, _, _, top_latents, bottom_latents = vqvae.encode(images)
            print("original top lat:", top_latents)

            top_min_value = torch.min(top_latents)
            bottom_min_value = torch.min(bottom_latents)
            top_max_value = torch.max(top_latents)
            bottom_max_value = torch.max(bottom_latents)

            top_latents = normalize(top_latents)
            bottom_latents = normalize(bottom_latents)
            print(" norm original top lat:", top_latents)
            top_var = Variable(top_latents)
            bottom_var = Variable(bottom_latents)

            top_output = top_model(top_var)
            bottom_output = bottom_model(bottom_var)
            print(top_output.shape)
            print(bottom_output.shape)
            top_output = sample_from_discretized_mix_logistic_1d(top_output, nr_logistic_mix)
            bottom_output = sample_from_discretized_mix_logistic_1d(bottom_output,nr_logistic_mix)
            print(top_output.shape)
            print(bottom_output.shape)
            print(top_output)

            top_output = transform_range(top_output, top_min_value, top_max_value)
            bottom_output = transform_range(bottom_output, bottom_min_value, bottom_max_value)
            print("genrated top lat: ", top_output.squeeze(1).long())

            gen_im = vqvae.decode_code(top_output.squeeze(1).long(),bottom_output.squeeze(1).long())
            #original_images = vqvae.decode_code(top_latents, bottom_latents)
    
            print("gen image ",gen_im)
            
            for i in range(gen_im.size(0)): 
                im = gen_im[i].permute(1, 2, 0)
                #print(im.shape)  # Transpose to H x W x C format
                min_value = torch.min(im)
                max_value = torch.max(im)
                normalized_input = (im - min_value) / (max_value - min_value)
                plt.figure(figsize=(6, 6))
                plt.imshow(normalized_input)
                plt.axis('off')
                plt.title(f"Image {i+1}")
                plt.show()

if __name__ == "__main__":
    main()
    #evaluate_vqvae()
    # vqvae = VQVAE()
    # vqvae.load_state_dict(torch.load('vqvae_epoch_10.pt', map_location=torch.device('cuda')))  # Load your trained model
    # vqvae.eval()
    # prova(vqvae)