import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from model import Generator, Discriminator
from dataset import CustomDataset
from loss import VGGLoss
import matplotlib.pyplot as plt
from utils import (
    clear_directories,
    save_checkpoint,
    load_checkpoint,
    debug_dataset,
    plot_training_losses,
    log_metrics_to_tensorboard,
    log_images_to_tensorboard,
    create_gif,
)

"""
This is a PyTorch settings that optimizes performance when using CuDNN.

What is does:
When benchmark = True, PyTorch dynamically finds the fastest convolution algorithms for your model based on the input size.
If your input size stays the same across iterations, CuDNN caches the best algorithm and speeds up training.

When to use it:
✅ Use it when input sizes are fixed (e.g., images of the same size).
❌ Avoid it when input sizes change frequently (e.g., variable-sized sequences in NLP), as it will constantly search for the best algorithm, causing overhead.
"""
torch.backends.cudnn.benchmarks = True


def prompt_for_epoch(model):
    """Prompts the user for an epoch number to load a model from."""
    epoch = input(f"What epoch do you want to load the {model} model from: ").strip()

    if not epoch.isdigit():
        print("Invalid input. Defaulting to epoch 0.")
        return 0

    return int(epoch)


def prepare_dataloader():
    """
    Prepares the DataLoader for training by loading the dataset with proper transformations.
    
    Returns:
        DataLoader: The DataLoader instance for training.
    """
    dataset = CustomDataset(root_dir=config.DATASET_DIR, processed_img_dir=config.PROCESSED_IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=config.NUM_WORKERS)

    return dataset, loader


def train_model():
    """
    Main function to train the SRGAN model. It initializes the generator and discriminator,
    defines the loss functions, sets up optimizers, and executes the training loop.
    """
    clear_directories()
    
    dataset, loader = prepare_dataloader()
    
    debug_dataset(dataset)
    
    generator = Generator(config.IN_CHANNELS).to(config.DEVICE)
    discriminator = Discriminator(config.IMG_CHANNELS).to(config.DEVICE)

    opt_gen = optim.Adam(generator.parameters(), lr=config.GEN_LEARNING_RATE, betas=(0.9, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.DISC_LEARNING_RATE, betas=(0.9, 0.999))

    vgg = VGGLoss()
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(f"{config.LOG_DIR}")

    if config.LOAD_MODEL:
        load_checkpoint("generator", prompt_for_epoch("generator"), generator, opt_gen)
        load_checkpoint("discriminator", prompt_for_epoch("discriminator"), discriminator, opt_disc)

    step = 0
    gen_losses, disc_losses = [], []
    generator.train(), discriminator.train()

    for epoch in range(config.EPOCHS_NUM):
        loop = tqdm(loader, leave=True)

        for batch_idx, (low_res, high_res) in enumerate(loop):
            low_res, high_res = low_res.to(config.DEVICE), high_res.to(config.DEVICE)

            """Discriminator Training"""

            fake = generator(low_res)
            
            disc_real = discriminator(high_res)
            disc_fake = discriminator(fake.detach())

            disc_real_loss = bce(disc_real, torch.ones_like(disc_real) * 0.9)
            disc_fake_loss = bce(disc_fake, torch.zeros_like(disc_fake))

            disc_loss = disc_real_loss + disc_fake_loss
            disc_losses.append(disc_loss.item())

            opt_disc.zero_grad()
            disc_loss.backward()
            opt_disc.step()

            """Generator Training"""

            disc_fake = discriminator(fake)
            adversarial_loss = config.ADVERSARIAL_LOSS * bce(disc_fake, torch.ones_like(disc_fake))
            fake = torch.nn.functional.interpolate(fake, size=high_res.shape[2:], mode='bilinear', align_corners=False)
            vgg_loss = 0.006 * vgg(fake, high_res)

            gen_loss = vgg_loss + adversarial_loss
            gen_losses.append(gen_loss.item())

            opt_gen.zero_grad()
            gen_loss.backward()
            opt_gen.step()

            # if batch_idx % 200 == 0:
            print(f"EPOCH[{epoch + 1}/{config.EPOCHS_NUM}], BATCH[{batch_idx}/{len(loader)}], "
                f"\nDISC REAL LOSS: {disc_real_loss:.2f}, DISC FAKE LOSS: {disc_fake_loss:.2f}, DISC LOSS: {disc_loss:.2f}, "
                f"\nVGG LOSS: {vgg_loss:.2f}, ADVERSARIAL LOSS: {adversarial_loss:.2f}, GEN LOSS: {gen_loss:.2f}")
            
            with torch.no_grad():
                log_metrics_to_tensorboard(writer, disc_loss.item(), gen_loss.item(), step)                
                log_images_to_tensorboard(writer, high_res, fake, step)

            step += 1

        if epoch % 10 == 0 and config.SAVE_MODEL:
            save_checkpoint("discriminator", epoch, discriminator, opt_disc)
            save_checkpoint("generator", epoch, generator, opt_gen)
            
    plot_training_losses(disc_losses, gen_losses)
    
    create_gif()


if __name__ == "__main__":
    train_model()