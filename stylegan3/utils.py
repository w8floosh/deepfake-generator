import PIL
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import torch

def print_generated(gen):
    img = (gen[0].permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    plt.imsave('img0.jpg', PIL.Image.fromarray(img.cpu().numpy(), "RGB"))
    plt.axis("off")
    plt.show()

def plot_avg_epoch_data(epochs, g_adv_loss, g_pix_loss, d_real_loss, d_fake_loss):
    import matplotlib.pyplot as plt
    import numpy as np

    # Sample data (replace with actual data)

    epochs_ticks = np.arange(1, epochs+1)

    # Create subplots
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot generator and discriminator losses
    ax1.plot(epochs_ticks, g_adv_loss, 'v--', color='tab:blue', label='Generator adversarial loss (BCE w/ logits)')
    ax1.plot(epochs_ticks, g_pix_loss, 'v--', color='tab:cyan', label='Generator pixelwise loss (L1)')
    ax1.plot(epochs_ticks, d_real_loss, 'o-', color='tab:green', label='Discriminator real loss (BCE w/ logits)' )
    ax1.plot(epochs_ticks, d_fake_loss, 'x-', color='tab:red', label='Discriminator fake loss (BCE w/ logits)')

    # Set labels, title, and legend for the first subplot
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('avg loss')
    ax1.set_title('Average StyleGAN3 losses per epoch')
    ax1.legend(loc='upper center')
    ax1.grid(True)

    ax1.set_xticks(range(1, epochs+1))
    ax1.set_xlim(1, epochs)

    # Save the plot to a file
    plt.savefig('stylegan3/gan_training_losses.png', dpi=300)
