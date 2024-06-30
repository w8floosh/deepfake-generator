import torch, pickle
from torch.cuda.amp import autocast, GradScaler 
from .utils import plot_avg_epoch_data, print_generated


def tune(
    dataset: torch.utils.data.Dataset,
    pickle,
    epochs,
    batch_size,
    truncation_psi=0.5,
    truncation_cutoff=8,
    accumulation_steps=8,
    batch_print_interval=None,
    checkpoint_interval=5,
    **learning_params
):
    generator = pickle["G_ema"]
    discriminator = pickle["D"]

    generator.train()
    discriminator.train()

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
        for i, imgs in enumerate(dataloader):
            # Adversarial ground truths
            print(f"E{epoch+1}B{i+1} - CURRENT LEARNING RATE - D: {optimizer_D.param_groups[0]['lr']}, G: {optimizer_G.param_groups[0]['lr']}")

            # ------------------
            # Train Generator
            # ------------------
            generator = generator.cuda()
            z = torch.randn([batch_size, generator.z_dim]).cuda()
            c = None
            gen_imgs = generator(
                z,
                c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                force_fp32=True
            ).cuda()

            valid = valid.cuda()
            fake = fake.cuda()

            discriminator = discriminator.cuda()
            a_loss = (
                adversarial_loss(discriminator(gen_imgs, c), valid) / accumulation_steps
            )

            real_imgs = imgs.cuda()
            p_loss = (pixelwise_loss(gen_imgs, real_imgs) / accumulation_steps)

            # Loss measures generator's ability to fool the discriminator

            g_loss = a_loss + p_loss

            scaler_G.scale(g_loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler_G.step(optimizer_G)
                scaler_G.update()
                optimizer_G.zero_grad()

            total_g_adv_loss += a_loss.item()
            total_g_pix_loss += p_loss.item()
            # ---------------------
            # Train Discriminator
            # ---------------------

            print(f"E{epoch+1}B{i+1} - Before releasing generator: {torch.cuda.mem_get_info()}")
            generator = generator.to("cpu")
            # Measure discriminator's ability to classify real from generated samples
            real_loss = (
                adversarial_loss(discriminator(real_imgs, c), valid)
                / accumulation_steps
            )
            total_d_real_loss += real_loss.item()

            fake_loss = (
                adversarial_loss(discriminator(gen_imgs.detach(), c), fake)
                / accumulation_steps
            )
            total_d_fake_loss += fake_loss.item()

            d_loss = (real_loss + fake_loss) / 2
            scaler_D.scale(d_loss).backward()

            if (i + 1) % accumulation_steps == 0:
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
                save(f"stylegan3/models/tuned_stylegan3_chk{epoch//checkpoint_interval}.pkl", pickle, generator, discriminator)

            
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
    return generator, discriminator


def save(path, pkl, generator, discriminator):
    pkl["G_ema"] = generator
    pkl["D"] = discriminator
    print("Saving the model")
    with open(path, 'wb') as file:
        pickle.dump(pkl, file)
    print("Finished saving the model")
