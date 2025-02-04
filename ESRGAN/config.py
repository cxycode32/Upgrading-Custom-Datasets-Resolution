import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_MODEL = True
LOAD_MODEL = False

IN_CHANNELS = 3
IMG_CHANNELS = 3

L1_LOSS = 1e-2
ADVERSARIAL_LOSS = 5e-3
GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 1e-4

BATCH_SIZE = 32
EPOCHS_NUM = 10
NUM_WORKERS = 4
LAMBDA_GP = 10

HIGH_RES = 256
LOW_RES = HIGH_RES // 4

MODELS_DIR = "models"
DATASET_DIR = "dataset/"
DEBUG_DIR = "debug"
ASSETS_DIR = "assets"
PROCESSED_IMAGE_DIR = "processed_images"
GENERATED_IMAGE_DIR = "generated_images"
LOG_DIR = "logs"
DIRECTORIES=[ASSETS_DIR, DEBUG_DIR, PROCESSED_IMAGE_DIR, GENERATED_IMAGE_DIR, LOG_DIR]
