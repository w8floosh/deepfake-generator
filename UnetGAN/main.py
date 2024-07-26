import argparse
from parse_logs import parse_gan_logs
import torch
import pickle
import utils
from torchvision import transforms
from dataset import CelebAHQDataset
from BigGAN import Generator, Unet_Discriminator
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler 
import PIL
from matplotlib import pyplot as plt
import numpy as np
from torch import Tensor
import os

paths = ["models", "datasets/celeba_hq_256", "D_ep_82.pth", "G_ep_82.pth", "G_ema_ep_82.pth"]
for p in paths:
    if not os.path.exists(p):  
        print("Il percorso non esiste")
        exit(0)

model_config = {
        'epoch_id': 'ep_82', 
        'unconditional': True, 
        'slow_mixup': True, 
        'consistency_loss': False, 
        'consistency_loss_and_augmentation': True, 
        'full_batch_mixup': True, 
        'debug': False, 
        'dataloader': 'celeba128', 
        'unet_mixup': True, 
        'progress_bar': False, 
        'display_mixed_batch': False, 
        'dataset': 'FFHQ', 
        'augment': False, 
        'num_workers': 4, 
        'pin_memory': True, 
        'shuffle': True, 
        'load_in_mem': False, 
        'use_multiepoch_sampler': False, 
        'model': 'BigGAN', 
        'D_param': 'SN', 
        'D_ch': 64, 
        'G_depth': 1, 
        'D_depth': 1, 
        'D_wide': True, 
        'G_shared': False, 
        'z_var': 1.0, 
        'hier': True, 
        'G_nl': 'relu', 
        'D_nl': 'relu', 
        'G_attn': '0', 
        'D_attn': '0', 
        'seed': 0, 
        'D_init': 'ortho', 
        'skip_init': True, 
        'G_lr': 0.0001, 
        'D_lr': 0.0005, 
        'D_B1': 0.0, 
        'D_B2': 0.999, 
        'batch_size': 20, 
        'G_batch_size': 0, 
        'num_G_accumulations': 1, 
        'num_D_steps': 1, 
        'num_D_accumulations': 1, 
        'split_D': False, 
        'num_epochs': 10000, 
        'parallel': True, 
        'D_fp16': False, 
        'D_mixed_precision': False, 
        'accumulate_stats': True, 
        'num_standing_accumulations': 100, 
        'G_eval_mode': True, 
        'save_every': 10000, 
        'num_save_copies': 1, 
        'num_best_copies': 2, 
        'which_best': 'FID', 
        'no_fid': False, 
        'test_every': 10000, 
        'num_inception_images': 50000, 
        'hashname': False, 
        'base_root': 'path/to/folder_for_results\\626687_ffhq_unet_bce_noatt_cutmix_consist', 
        'data_root': 'path/to/folder_for_results\\626687_ffhq_unet_bce_noatt_cutmix_consist/data', 
        'weights_root': 'path/to/folder_for_results\\626687_ffhq_unet_bce_noatt_cutmix_consist/weights', 
        'logs_root': 'path/to/folder_for_results\\626687_ffhq_unet_bce_noatt_cutmix_consist/logs', 
        'samples_root': 'path/to/folder_for_results\\626687_ffhq_unet_bce_noatt_cutmix_consist/samples', 
        'pbar': 'mine', 
        'name_suffix': '', 
        'experiment_name': '', 
        'config_from_name': False, 
        'ema': True, 
        'ema_decay': 0.9999, 
        'use_ema': True, 
        'ema_start': 21000, 
        'adam_eps': 1e-06, 
        'BN_eps': 1e-05, 
        'SN_eps': 1e-06, 
        'num_D_SVs': 1, 
        'num_D_SV_itrs': 1, 
        'G_ortho': 0.0, 
        'D_ortho': 0.0, 
        'toggle_grads': True, 
        'which_train_fn': 'GAN', 
        'load_weights': '', 
        'resume': True, 
        'logstyle': '%3.3e', 
        'log_G_spectra': False, 
        'log_D_spectra': False, 
        'sv_log_interval': 10, 
        'stop_it': 99999999999999, 
        'num_gpus': 2, 
        'random_number_string': '626687_ffhq_unet_bce_noatt_cutmix_consist', 
        'resolution': 256, 
        'n_classes': 1}


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
    ax1.set_title('Average UNETGAN losses per epoch')
    ax1.legend(loc='upper center')
    ax1.grid(True)

    ax1.set_xticks(range(1, epochs+1))
    ax1.set_xlim(1, epochs)

    # Save the plot to a file
    plt.savefig('UNETGAN_training_losses_15.png', dpi=300)

def save_evolution_generations():
    epochs = ["6","11","tuned"]
    path = "models"
    G_ema = Generator(**{**model_config, 'skip_init':True,
                                   'no_optim': True}).cuda()
    fixed_z, _ = utils.prepare_z_y(1, 128,
                                       model_config['n_classes'], device="cuda")
    fixed_z.sample_()
    for epoch in epochs:
        generator_path = os.path.join(path, f"G_ema_epoch_{epoch}.pth")
        weights = torch.load(generator_path, map_location=torch.device('cuda'))
        G_ema.load_state_dict(weights)
        G_ema.eval()
        with torch.no_grad():
            generated_images = G_ema(fixed_z)
        generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)
        for i, img in enumerate(generated_images):
            plt.imshow((img * 0.5 + 0.5).clip(0, 1))
            plt.axis('off')
            plt.savefig(f'generated_image_epoch_{epoch}.png')

def generate(n_images, batch_size, generator_path="G_ema_ep_82.pth"):
    from PIL import Image
    g_weights = torch.load(generator_path, map_location=torch.device('cuda'))
    G_ema = Generator(**{**model_config, 'skip_init':True,
                                   'no_optim': True}).cuda()
    G_ema.load_state_dict(g_weights)
    G_ema.eval()  
    n_batches = n_images // batch_size
    for batch in range(n_batches):
        # Prepare the latent vectors (z) for a batch of images
        fixed_z, _ = utils.prepare_z_y(batch_size, 128, model_config['n_classes'], device="cuda")
        fixed_z.sample_()

        # Generate images with the model
        with torch.no_grad():
            generated_images = G_ema(fixed_z, None)
        generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)

        # Save each generated image
        for i, img in enumerate(generated_images):
            img_norm = ((img * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)
            img_index = batch * batch_size + i + 1
            Image.fromarray(img_norm).save(f"generations/generated_image_{img_index}.jpg", "JPEG", quality=95)
    
    # fixed_z, _ = utils.prepare_z_y(n_images, 128,
    #                                    model_config['n_classes'], device="cuda")
    # fixed_z.sample_()

    # with torch.no_grad():
    #     generated_images = G_ema(fixed_z, None)
    # generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)
    # for i, img in enumerate(generated_images):
    #     img_norm = ((img * 0.5 + 0.5).clip(0, 1)*255).astype(np.uint8)
    #     # Image.fromarray(img).save(f"stylegan3/generated/generated_image_{batch*batch_size + i+1}.jpg", "JPEG", quality=95)
    #     Image.fromarray(img_norm).save(f"generations/generated_image_{i+1}.jpg", "JPEG", quality=95)


def tune( dataset: torch.utils.data.Dataset, generator, g_ema_path, discriminator, epochs, batch_size, accumulation_steps=8, batch_print_interval=None, checkpoint_interval=5, **learning_params):
    #device = "cpu"
    generator.train()
    generator = generator.cuda()
    discriminator.train()
    G_ema = Generator(**{**model_config, 'skip_init':True,
                                   'no_optim': True}).cuda()
    ema = utils.ema(generator, G_ema, model_config['ema_decay'], model_config['ema_start'])
    g_ema_weights = torch.load(g_ema_path, map_location=torch.device('cuda'))
    G_ema.load_state_dict(g_ema_weights)
    G_ema.train()
    G_ema = G_ema.cpu()
    # Define the loss functions and optimizers
    init_lr_g = learning_params.get("init_lr_g")
    init_lr_d = learning_params.get("init_lr_d")
    min_lr = learning_params.get("min_lr")
    lr_scale_factor = learning_params.get("lr_scale_factor")
    lr_patience = learning_params.get("lr_patience")

    adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
    pixelwise_loss = torch.nn.L1Loss().cuda()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=init_lr_g) # 0.0001 => 0.0000
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=init_lr_d)

    scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_G, mode='min', factor=lr_scale_factor, patience=lr_patience, min_lr=min_lr)
    scheduler_D = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_D, mode='min', factor=lr_scale_factor, patience=lr_patience, min_lr=min_lr)
    # # Initialize the dataloader for images
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    #Enable gradient calculation for the models
    for param in discriminator.parameters():
        param.requires_grad = True

    for param in generator.parameters():
        param.requires_grad = True

    valid = torch.ones((batch_size, 1))
    fake = torch.zeros((batch_size, 1))

    scaler_G = GradScaler()
    scaler_D = GradScaler()

    avg_g_adv_losses = []
    avg_g_pix_losses = []
    avg_d_real_losses = []
    avg_d_fake_losses = []

    for epoch in range(epochs):
        total_g_adv_loss = 0.0
        total_g_pix_loss = 0.0
        total_d_real_loss = 0.0
        total_d_fake_loss = 0.0
        itr = 0 
        for i, imgs in enumerate(dataloader):
            # Adversarial ground truths
            print(f"E{epoch+1}B{i+1} - CURRENT LEARNING RATE - D: {optimizer_D.param_groups[0]['lr']}, G: {optimizer_G.param_groups[0]['lr']}")
            itr += 1
            # ------------------
            # Train Generator
            # ------------------
            generator = generator.cuda()
            z, _ = utils.prepare_z_y(batch_size, 128,
                                       model_config['n_classes'], device="cuda")
            z.sample_()
            y = None
            gen_imgs = generator(z,y).cuda()

            valid = valid.cuda()
            fake = fake.cuda()

            discriminator = discriminator.cuda()
            _, output = discriminator(gen_imgs)
            a_loss = (
                adversarial_loss(output, valid) / accumulation_steps
            )

            real_imgs = imgs.cuda()
            p_loss = (pixelwise_loss(gen_imgs, real_imgs) / accumulation_steps)

            # Loss measures generator's ability to fool the discriminator

            g_loss = a_loss + p_loss

            scaler_G.scale(g_loss).backward()

            if (i + 1) % accumulation_steps == 0:
                #utils.ortho(generator, model_config['G_ortho'])
                scaler_G.step(optimizer_G)
                scaler_G.update()
                optimizer_G.zero_grad()

            total_g_adv_loss += a_loss.item()
            total_g_pix_loss += p_loss.item()
            G_ema = G_ema.cuda()
            ema.update(itr)
            G_ema = G_ema.cpu()
            # ---------------------
            # Train Discriminator
            # ---------------------

            print(f"E{epoch+1}B{i+1} - Before releasing generator: {torch.cuda.mem_get_info()}")
            generator = generator.to("cpu")
            # Measure discriminator's ability to classify real from generated samples
            _, output = discriminator(real_imgs)
            real_loss = (
                adversarial_loss(output, valid)
                / accumulation_steps
            )
            total_d_real_loss += real_loss.item()
            _, output = discriminator(gen_imgs.detach())
            fake_loss = (
                adversarial_loss(output, fake)
                / accumulation_steps
            )
            total_d_fake_loss += fake_loss.item()

            d_loss = (real_loss + fake_loss) / 2
            scaler_D.scale(d_loss).backward()

            if (i + 1) % accumulation_steps == 0:
                #utils.ortho(discriminator, model_config['D_ortho'])
                scaler_D.step(optimizer_D)
                scaler_D.update()
                optimizer_D.zero_grad()

            print(
                f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}]"
                f"[D loss: {d_loss.item():.4f} (real: {real_loss.item():.4f}, fake: {fake_loss.item():.4f})]"
                f"[G loss: {g_loss.item():.4f} (adversarial: {a_loss.item():.4f}, pixelwise: {p_loss.item():.4f})]"
            )

            if (
                batch_print_interval
                and (i + 1) * (epoch + 1) % batch_print_interval == 0
            ):
                print_generated(gen_imgs)
            
            print(f"E{epoch+1}B{i+1} - Before releasing discriminator: {torch.cuda.mem_get_info()}")
            discriminator = discriminator.to("cpu")
            
            if (epoch > 0 and epoch % checkpoint_interval == 0):
                #save(f"stylegan3/models/tuned_stylegan3_chk{epoch//checkpoint_interval}.pkl", pickle, generator, discriminator)
                torch.save(generator.state_dict(), f'models/G_epoch_{epoch}.pth')
                torch.save(discriminator.state_dict(), f'models/D_epoch_{epoch}.pth')
                torch.save(G_ema.state_dict(), f'models/G_ema_epoch_{epoch}.pth')

        avg_g_adv_losses.append(total_g_adv_loss / len(dataloader))
        avg_g_pix_losses.append(total_g_pix_loss / len(dataloader))
        avg_d_real_losses.append(total_d_real_loss / len(dataloader))
        avg_d_fake_losses.append(total_d_fake_loss / len(dataloader))

        scheduler_G.step(g_loss)
        scheduler_D.step(d_loss)

    plot_avg_epoch_data(
        epochs, 
        avg_g_adv_losses, 
        avg_g_pix_losses, 
        avg_d_real_losses, 
        avg_d_fake_losses
    )
            
    print("Finished fine-tuning")
    return generator, discriminator, G_ema

def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda")
    parser = argparse.ArgumentParser(description='Parameters of the model')
    parser.add_argument('--mode', choices=['train','generate', 'parse'], help='Specify whether to train model, generate images or parse model results')
    parser.add_argument('--resume', type=int, required=False)
    #parser.add_argument('--path', help='Specify the path of the model')
    args = parser.parse_args()

    epoch_resume = args.resume or None
    num_epochs = 15 - epoch_resume if epoch_resume else 15

    if args.mode == 'parse':
        avgs_d_real, avgs_d_fake, avgs_g_adv, avgs_g_pix = parse_gan_logs("logs.txt")
        plot_avg_epoch_data(
            num_epochs, 
            avgs_g_adv, 
            avgs_g_pix, 
            avgs_d_real, 
            avgs_d_fake
        )
        return
          


    D_path = f"models/D_epoch_{args.resume}.pth" if args.resume else "D_ep_82.pth"
    G_path = f"models/G_epoch_{args.resume}.pth" if args.resume else "G_ep_82.pth"
    G_ema_path = f"models/G_ema_epoch_{args.resume}.pth" if args.resume else "G_ema_ep_82.pth"
    d_weights = torch.load(D_path, map_location=torch.device('cuda'))
    g_weights = torch.load(G_path, map_location=torch.device('cuda'))

    G = Generator(**model_config)
    D = Unet_Discriminator(**model_config)
    
    G.load_state_dict(g_weights)
    D.load_state_dict(d_weights)
        
    if (args.mode =="generate"):
        batch_size = 16
        n_images = 10000
        imgs = generate(n_images, batch_size, generator_path=G_ema_path)
        print("gen")
    else:
        transform = transforms.Compose(
            [
                #transforms.Resize(model_config["resolution"]),
                transforms.CenterCrop(model_config["resolution"]),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        
        dataset = CelebAHQDataset(
            #img_dir="C:/Users/sabry/Downloads/Dataset/celeba_hq_256", transform=transform, num_images=10000
            img_dir="datasets/celeba_hq_256", transform=transform, num_images=10000
        )
        batch_size = 16
        tuned_generator, tuned_discriminator, tuned_G_ema= tune(
            dataset,
            G,
            G_ema_path,
            D,
            num_epochs,
            batch_size,
            init_lr_g=1e-05 if epoch_resume else 0.00001, 
            init_lr_d=1e-07 if epoch_resume else 0.000015, 
            min_lr=1e-7,
            lr_scale_factor=0.1,
            lr_patience=2,
        )
        torch.save(tuned_generator.state_dict(), 'models/G_tuned.pth')
        torch.save(tuned_discriminator.state_dict(), 'models/D_tuned.pth')
        torch.save(tuned_G_ema.state_dict(), 'models/G_ema_tuned.pth')

if __name__ == "__main__":
    main()