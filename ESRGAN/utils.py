import os
import glob
import torch
import shutil
import imageio
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import config


def clear_directories():
    """
    Deletes all directories specified in the configuration file.
    
    This is useful for clearing previous training outputs, ensuring
    that new experiments start fresh without leftover data.
    """
    for directory in config.DIRECTORIES:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            print(f"{directory}/ deleted successfully!")


def save_checkpoint(type, epoch, model, optimizer, dir=config.MODELS_DIR):
    """
    Saves the model and optimizer states as a checkpoint.

    Args:
        type (str): The type of model to save ('discriminator' or 'generator').
        epoch (int): The current epoch, used to name the checkpoint.
        model (torch.nn.Module): The model whose state needs to be saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state needs to be saved.
        dir (str, optional): Directory to store the checkpoint. Defaults to config.MODELS_DIR.
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


def load_checkpoint(type, epoch, model, optimizer, dir=config.MODELS_DIR, learning_rate=config.GEN_LEARNING_RATE if type=="generator" else config.DISC_LEARNING_RATE):
    """
    Loads a saved model checkpoint.

    Args:
        type (str): The type of model to load ('discriminator' or 'generator').
        epoch (int): The epoch from which the model should be loaded.
        model (torch.nn.Module): The model where the checkpoint is loaded.
        optimizer (torch.optim.Optimizer): The optimizer where the checkpoint is loaded.
        dir (str, optional): Directory where the checkpoint is stored. Defaults to config.MODELS_DIR.
        learning_rate (float, optional): Learning rate reset after loading the checkpoint.
            Defaults to config.GEN_LEARNING_RATE for the generator and config.DISC_LEARNING_RATE for the discriminator.

    Warning:
        If the checkpoint file does not exist, the function prints a warning and does not modify the model.
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


def custom_save_image(tensor_image, save_path):
    """
    Saves a tensor image to the specified path.
    
    Args:
        tensor_image (Tensor): The image tensor to save. Must have shape (C, H, W).
        save_path (str): Path to save the image file.
        
    The function ensures the tensor is within [0, 1] range and converts it to a PIL image before saving.
    """
    tensor_image = tensor_image.clamp(0, 1)
    pil_image = transforms.ToPILImage()(tensor_image)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pil_image.save(save_path)


def debug_dataset(dataset, num, dir=config.DEBUG_DIR):
    """
    Saves sample images from the dataset for debugging purposes.

    This function helps visualize both low-resolution and high-resolution images
    side by side, allowing you to inspect the dataset and ensure that the images
    are being processed correctly before training the model. It saves the images
    as PNG files in the specified directory.

    Args:
        dataset (iterable): The dataset to debug. Should be an iterable that yields
                             tuples of (low_res_image, high_res_image).
        num (int): The number of sample pairs to save from the dataset. This controls
                   how many (low_res, high_res) pairs will be saved. The function
                   will stop once this number is reached.
        dir (str, optional): The directory where the debug images will be saved.
                             Defaults to `config.DEBUG_DIR`. If the directory does
                             not exist, it will be created.

    Returns:
        None: This function saves images to the specified directory and prints
              out the paths where the images are saved.
    """
    os.makedirs(dir, exist_ok=True)
    
    for idx, (low_res, high_res) in enumerate(dataset):
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))

        axs[0].imshow(low_res.permute(1, 2, 0).numpy())
        axs[0].set_title("Low Resolution")
        axs[0].axis("off")

        axs[1].imshow(high_res.permute(1, 2, 0).numpy())
        axs[1].set_title("High Resolution")
        axs[1].axis("off")

        debug_path = os.path.join(dir, f"debug_sample_{idx:03d}.png")
        plt.savefig(debug_path, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved debug image: {debug_path}")

        if idx >= num:
            break


def save_generated_image(fake, epoch, batch_idx, dir=config.GENERATED_IMAGE_DIR):
    """
    Saves a generated image for visualization.
    
    Args:
        fake (torch.Tensor): Generated high-resolution image tensor.
        epoch (int): Current epoch.
        batch_idx (int): Current batch index.
        save_dir (str, optional): Directory to save images.
    """
    os.makedirs(dir, exist_ok=True)
    fake = fake.detach().cpu()
    fake = torch.clamp(fake, 0, 1)
    
    for i in range(fake.size(0)):
        img_path = os.path.join(dir, f"epoch_{epoch:03d}_batch_{batch_idx:03d}_img_{i:03d}.png")
        vutils.save_image(fake[i], img_path, normalize=False)
        print(f"Saved generated image: {img_path}")


def plot_training_losses(disc_losses, gen_losses, save_dir=config.ASSETS_DIR, filename="training_loss.png"):
    """
    Plots and saves the loss curves for the discriminator and generator.

    Args:
        disc_losses (list): List of discriminator loss values over training steps.
        gen_losses (list): List of generator loss values over training steps.
        save_dir (str, optional): Directory to save the plot. Defaults to config.ASSETS_DIR.
        filename (str, optional): Name of the saved file. Defaults to "training_loss.png".

    The function saves the plot and also displays it.
    """
    plt.figure(figsize=(10, 5))
    
    plt.plot(np.arange(len(disc_losses)), disc_losses, label="Discriminator Loss", alpha=0.7)
    plt.plot(np.arange(len(gen_losses)), gen_losses, label="Generator Loss", alpha=0.7)
    
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
    Logs loss values to TensorBoard for visualization.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        disc_loss (float): Discriminator loss at the current step.
        gen_loss (float): Generator loss at the current step.
        step (int): Training step number.

    TensorBoard helps in tracking training progress over time.
    """
    writer.add_scalar("Discriminator Loss", disc_loss, global_step=step)
    writer.add_scalar("Generator Loss", gen_loss, global_step=step)
    
    
def log_images_to_tensorboard(writer, real_images, fake_images, step):
    """
    Logs real and generated images to TensorBoard.

    Args:
        writer (SummaryWriter): TensorBoard writer instance.
        real_images (torch.Tensor): Batch of real images (shape: B, C, H, W).
        fake_images (torch.Tensor): Batch of generated images (shape: B, C, H, W).
        step (int): Training step number.

    Logs only the first 8 images from each batch for visualization.
    """
    grid_real = torchvision.utils.make_grid(real_images[:8], normalize=True)
    grid_fake = torchvision.utils.make_grid(fake_images[:8], normalize=True)

    writer.add_image("Real Images", grid_real, step)
    writer.add_image("Generated Images", grid_fake, step)


def create_gif(save_dir=config.ASSETS_DIR, image_dir=config.GENERATED_IMAGE_DIR, filename="gan_training"):
    """
    Creates a GIF from saved generated images.

    Args:
        save_dir (str, optional): Directory to save the GIF. Defaults to config.ASSETS_DIR.
        image_dir (str, optional): Directory containing the images. Defaults to config.GENERATED_IMAGE_DIR.
        filename (str, optional): Base name for the GIF file. Defaults to "gan_training".

    The function sorts images by modification time before creating the GIF.
    If no images are found, a warning is displayed.
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