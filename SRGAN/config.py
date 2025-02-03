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
PROCESSED_IMAGE_DIR = "processed_images"
GENERATED_IMAGE_DIR = "generated_images"
LOG_DIR = "logs"
DIRECTORIES=[ASSETS_DIR, PROCESSED_IMAGE_DIR, GENERATED_IMAGE_DIR, LOG_DIR]
