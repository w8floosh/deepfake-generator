import os
import pickle
import numpy as np
import torch

from stylegan3.utils import plot_avg_epoch_data
from . import training, generate
from torchvision import transforms
from .dataset import CelebAHQDataset
from argparse import ArgumentParser

num_epochs = 10
batch_size = 16
z_dim = 512
c_dim = 0  # No conditioning
truncation_psi = 0.5
truncation_cutoff = 8

parser = ArgumentParser(prog="stylegan3")

parser.add_argument(
    "-g",
    "--generate",
    type=int,
    help="Generate images through previously tuned StyleGAN3",
)

parser.add_argument(
    "-p",
    "--path",
    default="stylegan3/models/stylegan3-t-ffhqu-256x256.pkl",
    help="StyleGAN3 model path",
)

args = parser.parse_args()

with open(args.path, "rb") as f:
    # checkpoint = legacy.load_network_pkl(f)
    try:
        checkpoint = pickle.load(f)
    except:
        checkpoint = torch.load(f)

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
