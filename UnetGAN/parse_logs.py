import re
import numpy as np

# Function to parse the log file and calculate averages
def parse_gan_logs(file_path):
    with open(file_path, 'r') as file:
        logs = file.readlines()
    
    epoch_data = {}
    current_epoch = None

    for line in logs:
        if "CURRENT LEARNING RATE" in line:
            # Parse the learning rates (D and G)
            match = re.search(r'E(\d+)B\d+ - CURRENT LEARNING RATE - D: ([\d\.e-]+), G: ([\d\.e-]+)', line)
            if match:
                epoch = int(match.group(1))
                if epoch == 16: break
                d_lr = float(match.group(2))
                g_lr = float(match.group(3))
                if epoch not in epoch_data:
                    epoch_data[epoch] = {
                        'd_losses_real': [],
                        'd_losses_fake': [],
                        'g_losses_adversarial': [],
                        'g_losses_pixelwise': [],
                    }
                current_epoch = epoch

        elif "D loss" in line:
            # Parse the losses
            match = re.search(
                r'\[Epoch \d+/\d+\] \[Batch \d+/\d+\]\[D loss: ([\d\.e-]+) \(real: ([\d\.e-]+), fake: ([\d\.e-]+)\)\]\[G loss: ([\d\.e-]+) \(adversarial: ([\d\.e-]+), pixelwise: ([\d\.e-]+)\)\]', line)
            if match:
                print(line)
                d_loss = float(match.group(1))
                d_loss_real = float(match.group(2))
                d_loss_fake = float(match.group(3))
                g_loss = float(match.group(4))
                g_loss_adversarial = float(match.group(5))
                g_loss_pixelwise = float(match.group(6))

                if current_epoch is not None:
                    epoch_data[current_epoch]['d_losses_real'].append(d_loss_real)
                    epoch_data[current_epoch]['d_losses_fake'].append(d_loss_fake)
                    epoch_data[current_epoch]['g_losses_adversarial'].append(g_loss_adversarial)
                    epoch_data[current_epoch]['g_losses_pixelwise'].append(g_loss_pixelwise)

    avgs_d_real = []
    avgs_d_fake = []
    avgs_g_adv = []
    avgs_g_pix = []
    # Calculate average losses per epoch
    for epoch, data in epoch_data.items():
        # print(data)
        avg_d_real = np.mean(data['d_losses_real'])
        avg_d_fake = np.mean(data['d_losses_fake'])
        avg_g_adversarial = np.mean(data['g_losses_adversarial'])
        avg_g_pixelwise = np.mean(data['g_losses_pixelwise'])
        
        avgs_d_real.append(avg_d_real)
        avgs_d_fake.append(avg_d_fake)
        avgs_g_adv.append(avg_g_adversarial)
        avgs_g_pix.append(avg_g_pixelwise)
        
        # print(f"Epoch {epoch}:")
        # print(f"  D Loss Real (avg): {avg_d_real:.4f}")
        # print(f"  D Loss Fake (avg): {avg_d_fake:.4f}")
        # print(f"  G Loss Adversarial (avg): {avg_g_adversarial:.4f}")
        # print(f"  G Loss Pixelwise (avg): {avg_g_pixelwise:.4f}")

    return avgs_d_real, avgs_d_fake, avgs_g_adv, avgs_g_pix