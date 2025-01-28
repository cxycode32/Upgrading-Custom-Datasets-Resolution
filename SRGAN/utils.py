import os
import cv2
import glob
import torch
import shutil
import imageio
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import config


def clear_directories():
    """
    Deletes all directories specified in the configuration file.
    This is useful for clearing previous training outputs.
    """
    for directory in config.DIRECTORIES:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"{directory}/ deleted successfully!")


def save_checkpoint(type, epoch, model, optimizer, dir=config.MODEL_DIR):
    """
    Saves the model and optimizer states as a checkpoint.

    Args:
        type (str): The type of model to save ('critic' or 'generator').
        epoch (int): The current epoch, used to name the checkpoint.
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save states.
        dir (str, optional): Directory to store the checkpoint. Defaults to config.MODEL_DIR.
    """
    print("Saving checkpoint......")
    os.makedirs(dir, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filepath = f"{dir}/{type}_{epoch}.pth"
    torch.save(checkpoint, filepath)
    print("Checkpoint saved successfully.")


def load_checkpoint(type, epoch, model, optimizer, dir=config.MODEL_DIR, learning_rate=config.GEN_LEARNING_RATE if type=="generator" else config.DISC_LEARNING_RATE):
    """
    Loads a saved model checkpoint.

    Args:
        type (str): The type of model to load ('critic' or 'generator').
        epoch (int): Load the model from which epoch.
        model (torch.nn.Module): The model to load the checkpoint into.
        optimizer (torch.optim.Optimizer): The optimizer to restore states from the checkpoint.
        dir (str, optional): Directory where the checkpoint is stored. Defaults to config.MODEL_DIR.
        learning_rate (float, optional): Learning rate to be set after loading the optimizer state. Defaults to config.LEARNING_RATE.
    """
    checkpoint_path = os.path.join(dir, f"{type}_{epoch}.pth")

    if not os.path.isfile(checkpoint_path):
        print(f"Warning: Checkpoint file '{checkpoint_path}' not found. Falling back without loading checkpoint.")
        return

    print("Loading checkpoint......")
    checkpoint = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate
    print("Checkpoint loaded successfully.")


def save_progress_image(epoch, batch_idx, low_res, fake, high_res, class_name, img_name, root_dir=config.IMAGE_DIR):
    """
    Save progress images for visualization.

    Args:
        epoch (int): Current training epoch.
        batch_idx (int): Current batch index.
        low_res (tensor): Low-resolution input image.
        fake (tensor): Generated high-resolution image.
        high_res (tensor): Ground truth high-resolution image.
        class_name (str): The folder name of the class.
        img_name (str): The original filename of the image.
    """
    save_dir = os.path.join(root_dir, class_name)
    os.makedirs(save_dir, exist_ok=True)

    transform = transforms.ToPILImage()

    # Convert tensor to PIL Image
    fake_img = transform(fake.cpu())
    low_res_img = transform(low_res.cpu())
    high_res_img = transform(high_res.cpu())

    # Save images
    fake_img.save(f"{save_dir}/{img_name}_fake_epoch{epoch}_batch{batch_idx}.png")
    low_res_img.save(f"{save_dir}/{img_name}_low_res_epoch{epoch}_batch{batch_idx}.png")
    high_res_img.save(f"{save_dir}/{img_name}_high_res_epoch{epoch}_batch{batch_idx}.png")


def plot_training_losses(disc_losses, gen_losses, save_dir=config.ASSETS_DIR, filename="training_loss.png"):
    """
    Plots and saves the loss curves for the critic and generator during training.

    Args:
        disc_losses (list): List of discriminator loss values over epochs.
        generator_losses (list): List of generator loss values over epochs.
        save_dir (str, optional): Directory where plots will be saved. Defaults to config.ASSETS_DIR.
        filename (str, optional): Base name of the saved plot file. Defaults to "training_loss.png".
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(np.arange(len(disc_losses)), disc_losses, label="Discriminator Loss", alpha=0.7)
    plt.plot(np.arange(len(gen_losses)), gen_losses, label="Generator Loss", alpha=0.7)
    
    # plt.plot(disc_losses, label="Discriminator Loss")
    # plt.plot(gen_losses, label="Generator Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.legend()
    plt.grid()
    plt.title("SRGAN Training Loss")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/{filename}")

    plt.show()


def log_metrics_to_tensorboard(writer, disc_loss, gen_loss, step):
    """
    Logs loss values to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        disc_loss (float): Current discriminator loss value.
        gen_loss (float): Current generator loss value.
        step (int): Current training step for logging.
    """
    writer.add_scalar("Discriminator Loss", disc_loss, global_step=step)
    writer.add_scalar("Generator Loss", gen_loss, global_step=step)
    
    
def log_images_to_tensorboard(writer, real_images, fake_images, step):
    """
    Logs real and fake images to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        real_images (???): Real images.
        fake_images (???): Fake images.
        step (int): Current training step for logging.
    """
    grid_real = torchvision.utils.make_grid(real_images[:8], normalize=True)
    grid_fake = torchvision.utils.make_grid(fake_images[:8], normalize=True)

    writer.add_image("Real Images", grid_real, step)
    writer.add_image("Generated Images", grid_fake, step)


def create_gif(save_dir=config.ASSETS_DIR, image_dir=config.IMAGE_DIR, filename="gan_training"):
    """
    Creates a GIF from the saved sample images.

    Args:
        save_dir (str, optional): Directory to save the generated GIF. Defaults to config.ASSETS_DIR.
        image_dir (str, optional): Directory containing the images. Defaults to config.IMAGE_DIR.
        filename (str, optional): Base name for the GIF file. Defaults to "gan_training".
    """
    if not os.path.exists(image_dir):
        print(f"Warning: No images found in {image_dir}.")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    images = sorted(glob.glob(f"{image_dir}/*.png"), key=os.path.getmtime)
    if not images:
        print(f"Warning: No PNG images in {image_dir}. Skipping GIF creation.")
        return

    gif_images = [imageio.imread(img) for img in images]
    gif_path = os.path.join(save_dir, f"{filename}.gif")
    imageio.mimsave(gif_path, gif_images, fps=5)
    print(f"GIF saved at: {gif_path}")