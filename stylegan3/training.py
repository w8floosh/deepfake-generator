import torch
from torch.cuda.amp import autocast
from .utils import print_generated


def tune(
    dataset: torch.utils.data.Dataset,
    generator,
    discriminator,
    epochs,
    batch_size,
    truncation_psi=0.5,
    truncation_cutoff=8,
    accumulation_steps=8,
    batch_print_interval=None,
):
    # Define the loss functions and optimizers
    adversarial_loss = torch.nn.BCEWithLogitsLoss().cuda()
    pixelwise_loss = torch.nn.L1Loss().to()
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    # # Initialize the dataloader for images
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    # Enable gradient calculation for the models
    # for param in discriminator.parameters():
    #     param.requires_grad = True

    # for param in generator.parameters():
    #     param.requires_grad = True

    valid = torch.ones((batch_size, 1)).cuda()
    fake = torch.zeros((batch_size, 1)).cuda()

    for epoch in range(epochs):
        print("starting epoch", epoch)
        for i, imgs in enumerate(dataloader):
            print("starting batch", i)
            real_imgs = imgs.cuda()

            # Adversarial ground truths

            # ------------------
            # Train Generator
            # ------------------

            # Sample noise (latent vector z) as generator input

            # Generate a batch of images

            z = torch.randn([batch_size, generator.z_dim], device="cuda")
            c = None
            print(z)
            # for obj in generator.mapping:
            #     if hasattr(obj, "bias"):
            #         obj["bias"] = torch.nn.Parameter(obj["bias"].to(torch.float16))

            # for obj in discriminator:
            #     if hasattr(obj, "conv0"):
            #         obj["conv0"]["bias"] = torch.nn.Parameter(
            #             obj["conv0"]["bias"].to(torch.float16)
            #         )
            #     if hasattr(obj, "conv1"):
            #         obj["conv1"]["bias"] = torch.nn.Parameter(
            #             obj["conv0"]["bias"].to(torch.float16)
            #         )

            # with autocast(dtype=torch.float16):
            gen_imgs = generator(
                z,
                c,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                force_fp32=True,
            ).cuda()

            print(
                f"E{epoch}B{i}: Generated {batch_size} images of shape {gen_imgs.shape}"
            )

            a_loss = (
                adversarial_loss(discriminator(gen_imgs, c), valid) / accumulation_steps
            )
            p_loss = pixelwise_loss(gen_imgs, real_imgs) / accumulation_steps

            # Loss measures generator's ability to fool the discriminator
            g_loss = a_loss + p_loss

            g_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

            # ---------------------
            # Train Discriminator
            # ---------------------

            # Measure discriminator's ability to classify real from generated samples
            real_loss = (
                adversarial_loss(discriminator(real_imgs, c), valid)
                / accumulation_steps
            )
            fake_loss = (
                adversarial_loss(discriminator(gen_imgs.detach(), c), fake)
                / accumulation_steps
            )
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_D.step()
                optimizer_D.zero_grad()

            print(
                f"[Epoch {epoch+1}/{epochs}] [Batch {i+1}/{len(dataloader)}]"
                f"[D loss: {d_loss.item():.4f} (real: {real_loss.item():.4f}, fake: {fake_loss.item():.4f})]"
                f"[G loss: {g_loss.item():.4f} (adversarial: {a_loss.item():.4f}, pixelwise: {p_loss.item():.4f}]"
            )

            if (
                batch_print_interval
                and (i + 1) * (epoch + 1) % batch_print_interval == 0
            ):
                print_generated(gen_imgs)

    print("Finished fine-tuning")
    return generator.state_dict(), discriminator.state_dict()


def save(path, generator, discriminator):
    print("Saving the model")
    torch.save(
        {"generator": generator, "discriminator": discriminator},
        path,
    )
    print("Finished saving the model")
