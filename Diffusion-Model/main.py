import argparse
import torch
import pickle
import utils
from torchvision import transforms
from dataset import CelebAHQDataset
from BigGAN import Generator, Unet_Discriminator

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    parser = argparse.ArgumentParser(description='Parameters of the model')
    #parser.add_argument('--mode', choices=['train','generate'], help='Specify whether to train or generate model results')
    #parser.add_argument('--model', choices=['vqvae', 'pixelcnn'],help='Specify the model')
    parser.add_argument('--path',help='Specify the path of the model')
    args = parser.parse_args()

    num_epochs = 10
    batch_size = 16
    z_dim = 512
    c_dim = 0  # No conditioning
    truncation_psi = 0.5
    truncation_cutoff = 8

    d_weights = torch.load(args.path + "/D_ep_82.pth", map_location=torch.device('cpu'))
    d_optim_weights = torch.load(args.path + "/D_optim_ep_82.pth", map_location=torch.device('cpu'))
    g_weights = torch.load(args.path + "/G_ep_82.pth", map_location=torch.device('cpu'))

    G = Generator(BN_eps=1e-5, SN_eps=1e-6, G_lr=1e-4, hier = True, G_attn = "0").to(device)
    D = Unet_Discriminator(D_attn="0", D_lr=5e-4).to(device)
    
    G.load_state_dict(g_weights)
    #discriminator.load_state_dict(d_weights)
    #discriminator.optim.load_state_dict(d_optim_weights)
    print("ciao")
    print(G)
    exit(0)
    print("ciao")
    print(discriminator)
    print(generator)
    exit(0)
    if (args.generate):
        imgs = generate.generate_images(checkpoint, args.generate, 512, truncation_psi, truncation_cutoff)

    else:
        # Define the transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        # Initialize the datasets
        print("Loading training and validation dataset...")
        dataset = CelebAHQDataset(
            img_dir="stylegan3/datasets/celeba_hq_256", transform=transform, num_images=10000
        )
        # validation = CelebAHQDataset(
        #     img_dir="stylegan3/datasets/celeba_hq_256", transform=transform, num_images=3072, index=6928
        # )

        # Tuning phase
        tuned_generator, tuned_discriminator = training.tune(
            dataset,
            checkpoint,
            num_epochs,
            batch_size,
            truncation_psi,
            truncation_cutoff,
            init_lr_g=0.00001, 
            init_lr_d=0.000015, 
            min_lr=1e-7, 
            lr_scale_factor=0.1,
            lr_patience=2,
        )

        # Saving the tuned model
        training.save(
            "stylegan3/models/tuned_stylegan3.pkl", checkpoint, tuned_generator, tuned_discriminator
        )

if __name__ == "__main__":
    main()