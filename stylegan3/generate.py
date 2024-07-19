import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def generate_images(pickle, n, latent_dim, truncation_psi, truncation_cutoff, device='cuda'):
    """
    Generates n images from the GAN generator.

    Parameters:
    - generator: The trained GAN generator model
    - n: Number of images to generate
    - latent_dim: Dimension of the latent noise vector
    - device: Device to run the generation on ('cuda' or 'cpu')
    
    Returns:
    - images: A list of generated images
    """
    generator = pickle["G"].to(device)
    generator.eval()
    batch_size = 8
    ng = 0
    for batch in range(np.int32(np.ceil(n / batch_size))):
        print(f'Generated {ng} images')

    # Generate latent noise vectors
        noise = torch.randn(batch_size, latent_dim).to(device)
        
        # Generate images
        with torch.no_grad():
            # generated_images = generator(noise, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, force_fp32=True)
            generated_images = generator(noise, None, truncation_psi=truncation_psi, force_fp32=True)
        
        # Move images to CPU and convert to numpy array for visualization
        generated_images = generated_images.cpu().numpy()
        
        # Normalize images to [0, 1] range if necessary
        generated_images = (generated_images + 1) / 2
        
        for i, img in enumerate(generated_images):
            if ng != 0 and ng / n == 0:
                return
            ng += 1
            # Convert image format from (C, H, W) to (H, W, C)
            img = np.transpose(img, (1, 2, 0))
            # Convert to uint8
            img = (img * 255).astype(np.uint8)
            # Save image
            Image.fromarray(img).save(f"stylegan3/generated/generated_image_{batch*batch_size + i+1}.jpg", "JPEG", quality=95)
            

def plot_images(images, n):
    """
    Plots n generated images.

    Parameters:
    - images: List or array of generated images
    - n: Number of images to plot
    """
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i].transpose(1, 2, 0))
        plt.axis('off')
    plt.show()

# Example usage:
# Assuming you have a trained generator and its latent dimension
# generator = ...  # Your trained generator
# latent_dim = 100  # Example latent dimension

# Generate and plot 5 images
# generated_images = generate_images(generator, n=5, latent_dim=latent_dim, device='cuda')
# plot_images(generated_images, n=5)
