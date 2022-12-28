import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:2" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/"
VAL_DIR = "data/val"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_A = "human2anime.ckpt"
CHECKPOINT_GEN_B = "anime2human.ckpt"
CHECKPOINT_CRITIC_A = "disc_a.ckpt"
CHECKPOINT_CRITIC_B = "disc_b.ckpt"
WHICH_MODEL = ""
MODEL_DIR = "model/"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
