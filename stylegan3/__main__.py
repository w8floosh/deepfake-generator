import pickle
import torch
from . import training
from torchvision import transforms
from .dataset import CelebAHQDataset

# Define the transformations
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

# Initialize the datasets
print("Loading training and validation dataset...")
tuning = CelebAHQDataset(
    img_dir="datasets/celeba_hq_256", transform=transform, num_images=1000
)
validation = CelebAHQDataset(
    img_dir="datasets/celeba_hq_256", transform=transform, num_images=3000, index=7000
)

# Loading custom pre-trained model
model_path = "stylegan3/models/stylegan3-t-ffhqu-256x256.pkl"
with open(model_path, "rb") as f:
    # checkpoint = legacy.load_network_pkl(f)
    checkpoint = pickle.load(f)

# Extracting generator with EMA weights and discriminator
generator = checkpoint["G"].cuda()
discriminator = checkpoint["D"].cuda()

# Choose between evaluation or training mode
generator.train()
discriminator.train()

# Move the models to the GPU or keep them on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

# Hyperparameters
num_epochs = 5
batch_size = 16
z_dim = generator.z_dim
c_dim = 0  # No conditioning
truncation_psi = 0.5
truncation_cutoff = 8

# Tuning phase
tuned_generator, tuned_discriminator = training.tune(
    tuning,
    generator,
    discriminator,
    num_epochs,
    batch_size,
    truncation_psi,
    truncation_cutoff,
)

# Saving the tuned model
training.save(
    "stylegan3/models/tuned_generator.pkl", tuned_generator, tuned_discriminator
)
