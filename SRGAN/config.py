import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL = True
LOAD_MODEL = False

IN_CHANNELS = 3
IMG_CHANNELS = 3

ADVERSARIAL_LOSS = 1e-2
GEN_LEARNING_RATE = 1e-3
DISC_LEARNING_RATE = 1e-4

BATCH_SIZE = 32
EPOCHS_NUM = 100
NUM_WORKERS = 4

HIGH_RES = 96
LOW_RES = HIGH_RES // 4

MODEL_DIR = "models"
DATASET_DIR = "dataset/"
ASSETS_DIR = "assets"
IMAGE_DIR = "generated_images"
LOG_DIR = "logs"
DIRECTORIES=[ASSETS_DIR, IMAGE_DIR, LOG_DIR]

general_transform = A.Compose(
    [
        A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]
)

high_res_transform = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

low_res_transform = A.Compose(
    [
        A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ]
)