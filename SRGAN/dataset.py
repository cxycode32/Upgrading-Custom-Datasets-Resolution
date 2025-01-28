import os
import config
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset class for loading and processing images for SRGAN training.
    
    Args:
        root_dir (str): Path to the root directory containing image subdirectories.
    """
    def __init__(self, root_dir):
        super(CustomDataset, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            class_path = os.path.join(root_dir, name)
            files = os.listdir(class_path)
            self.data += list(zip(files, [index] * len(files)))
            
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
        img_file, label = self.data[index]
        root_and_dir = os.path.join(self.root_dir, self.class_names[label])
        
        image = Image.open(os.path.join(root_and_dir, img_file)).convert("RGB")
        image = np.array(image).astype(np.uint8)
        
        image = config.general_transform(image=image)["image"]
        low_res = config.low_res_transform(image=image)["image"]
        high_res = config.high_res_transform(image=image)["image"]
        
        return low_res, high_res
