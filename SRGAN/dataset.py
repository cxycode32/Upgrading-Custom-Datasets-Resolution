import os
from glob import glob
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from utils import save_image


class CustomDataset(Dataset):
    """
    Custom dataset class for loading and processing images for SRGAN training.
    
    Args:
        root_dir (str): Path to the root directory containing image subdirectories.
    """
    def __init__(self, root_dir, processed_img_dir):
        super(CustomDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.processed_img_dir = processed_img_dir
        self.class_names = os.listdir(root_dir)

        for class_name in self.class_names:
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path): 
                continue
            
            image_files = glob(os.path.join(class_path, "*"))
            self.data += [(image_path, class_name) for image_path in image_files]
            
            processed_class_path = os.path.join(processed_img_dir, class_name)
            os.makedirs(os.path.join(processed_class_path, "lr"), exist_ok=True)
            os.makedirs(os.path.join(processed_class_path, "hr"), exist_ok=True)
            
    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Loads an image, applies transformations, and returns the low-resolution and high-resolution versions.
        
        Args:
            index (int): Index of the image in the dataset.
        
        Returns:
            tuple: (low_res, high_res), transformed low and high-resolution images.
        """
        image_path, class_name = self.data[index]
        image_name = os.path.basename(image_path)

        print(f"{index}: {image_path}, {class_name}, {image_name}")
        
        image = Image.open(image_path).convert("RGB")
        
        low_res = process_to_low_res(image)
        high_res = process_to_high_res(image)
        
        self.save_image(low_res, class_name, image_name, "lr")
        self.save_image(high_res, class_name, image_name, "hr")
        
        return low_res, high_res
    
    def save_image(self, tensor_image, class_name, image_name, res_type):
        """
        Saves the image tensor to the appropriate directory (lr or hr).
        
        Args:
            tensor_image (Tensor): The image tensor to save.
            class_name (str): The class of the image (e.g., gpay).
            image_name (str): The name of the image file.
            res_type (str): The resolution type, either 'lr' or 'hr'.
        """
        save_dir = os.path.join(self.processed_img_dir, class_name, res_type)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, image_name)
        save_image(tensor_image, save_path)
    

def resize_and_pad(image, target_size, fill_color=(0, 0, 0)):
    """
    Resizes an image while maintaining aspect ratio and pads the remaining space with a fill color.

    Args:
        image (PIL.Image): Input image.
        target_size (tuple): (height, width) of the target size.
        fill_color (tuple): RGB color to fill the padding (default: black).

    Returns:
        PIL.Image: Resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size

    # Calculate the new size while keeping aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with the target size and fill color
    new_image = Image.new("RGB", (target_width, target_height), fill_color)

    # Paste the resized image onto the center of the new image
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(image, (paste_x, paste_y))

    return new_image


def process_to_low_res(image):
    image = resize_and_pad(image, (128, 128), fill_color=(0, 0, 0))  # Black padding
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 1.0)),
        transforms.ToTensor(),
    ])
    return transform(image)


def process_to_high_res(image):
    image = resize_and_pad(image, (256, 256), fill_color=(0, 0, 0))  # Black padding
    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.adjust_contrast(img, 1.5)),
        transforms.ToTensor(),
    ])
    return transform(image)
