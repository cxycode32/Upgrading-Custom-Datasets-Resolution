import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import config
from dataset import CustomDataset
from model import Generator, Discriminator, initialize_weights
from loss import VGGLoss
from utils import (
    clear_directories,
    save_checkpoint,
    load_checkpoint,
    debug_dataset,
    save_generated_image,
    plot_training_losses,
    log_metrics_to_tensorboard,
    log_images_to_tensorboard,
    create_gif,
)

"""
This is a PyTorch settings that optimizes performance when using CuDNN.

What it does:
- When benchmark = True, PyTorch dynamically finds the fastest convolution algorithms for your model based on the input size.
- If your input size stays the same across iterations, CuDNN caches the best algorithm and speeds up training.

When to use it:
✅ Use it when input sizes are fixed (e.g., images of the same size).
❌ Avoid it when input sizes change frequently (e.g., variable-sized sequences in NLP), as it will constantly search for the best algorithm, causing overhead.
"""
torch.backends.cudnn.benchmarks = True


def prompt_for_epoch(model):
    """
    Prompts the user for an epoch number to load a saved model.

    Args:
        model (str): The model type (e.g., 'generator' or 'discriminator').
        
    Returns:
        epoch (int): The selected epoch number. Defaults to 0 if input is invalid.
    """
    epoch = input(f"What epoch do you want to load the {model} model from: ").strip()

    if not epoch.isdigit():
        print("Invalid input. Defaulting to epoch 0.")
        return 0

    return int(epoch)


def prepare_dataloader():
    """
    Loads and prepares the dataset for training.

    Returns:
        tuple: (Dataset instance, DataLoader instance)
    """
    dataset = CustomDataset(dataset_dir=config.DATASET_DIR, processed_dataset_dir=config.PROCESSED_IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=config.NUM_WORKERS)

    return dataset, loader


def gradient_penalty(critic, real, fake, device=config.DEVICE):
    """
    Computes the gradient penalty for the Wasserstein GAN (WGAN) loss function.
    
    This enforces the Lipschitz constraint by calculating the norm of the gradients 
    with respect to interpolated images between real and fake data.

    Args:
        critic (nn.Module): The discriminator (or critic) of the WGAN.
        real (torch.Tensor): The real image batch.
        fake (torch.Tensor): The generated (fake) image batch.
        device (str): The device on which computations are performed (default: 'cuda').

    Returns:
        torch.Tensor: The computed gradient penalty.
    """
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)  # Generate random alpha
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)  # Interpolate between real and fake
    interpolated_images.requires_grad_(True)

    # Get the discriminator (critic) score for the interpolated images
    mixed_scores = critic(interpolated_images)

    # Compute the gradients of the critic with respect to the interpolated images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)  # Flatten gradient tensor
    gradient_norm = gradient.norm(2, dim=1)  # L2 norm of the gradient
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)  # Calculate the penalty (WGAN-GP)
    return gradient_penalty


def train_model():
    """
    Trains the model by handling:
    - Model initialization (Generator & Discriminator)
    - Loss function setup (Adversarial, Perceptual, MSE)
    - Optimizer setup (Adam for both models)
    - Training loop execution
    - Periodic checkpoint saving and TensorBoard logging
    """
    clear_directories()
    
    dataset, loader = prepare_dataloader()
    
    debug_dataset(dataset, num=10)
    
    generator = Generator(config.IN_CHANNELS).to(config.DEVICE)
    discriminator = Discriminator(config.IN_CHANNELS).to(config.DEVICE)

    initialize_weights(generator)

    opt_gen = optim.Adam(generator.parameters(), lr=config.GEN_LEARNING_RATE, betas=(0.9, 0.9))
    opt_disc = optim.Adam(discriminator.parameters(), lr=config.DISC_LEARNING_RATE, betas=(0.9, 0.9))

    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_disc = torch.cuda.amp.GradScaler()

    vgg = VGGLoss()
    l1 = nn.L1Loss()

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

            with torch.cuda.amp.autocast():
                fake = generator(low_res)
            
                disc_real = discriminator(high_res)
                disc_fake = discriminator(fake.detach())
                
                gp = gradient_penalty(discriminator, high_res, fake)

                disc_loss = (-(torch.mean(disc_real) - torch.mean(disc_fake)) + config.LAMBDA_GP * gp) 
                disc_losses.append(disc_loss.item())

            opt_disc.zero_grad()
            scaler_disc.scale(disc_loss).backward()
            scaler_disc.step(opt_disc)
            scaler_disc.update()

            """Generator Training"""

            with torch.cuda.amp.autocast():
                l1_loss = config.L1_LOSS * l1(fake, high_res)
                adversarial_loss = config.ADVERSARIAL_LOSS * -torch.mean(discriminator(fake))
                vgg_loss = vgg(fake, high_res)
                gen_loss = l1_loss + vgg_loss + adversarial_loss
                gen_losses.append(gen_loss.item())

            opt_gen.zero_grad()
            scaler_gen.scale(gen_loss).backward()
            scaler_gen.step(opt_gen)
            scaler_gen.update()

            log_metrics_to_tensorboard(writer, disc_loss.item(), gen_loss.item(), step)
            print(f"EPOCH[{epoch + 1}/{config.EPOCHS_NUM}], BATCH[{batch_idx + 1}/{len(loader)}], "
                f"DISC LOSS: {disc_loss:.2f}, GEN LOSS: {gen_loss:.2f}")
            
            step += 1
            
            loop.set_postfix(
                gp=gp.item(),
                critic=disc_loss.item(),
                l1=l1_loss.item(),
                vgg=vgg_loss.item(),
                adversarial=adversarial_loss.item(),
            )

        if epoch % 10 == 0 and config.SAVE_MODEL:
            save_checkpoint("discriminator", epoch, discriminator, opt_disc)
            save_checkpoint("generator", epoch, generator, opt_gen)
            
    plot_training_losses(disc_losses, gen_losses)
    
    create_gif()


if __name__ == "__main__":
    train_model()