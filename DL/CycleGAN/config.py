import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "dataset/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 1
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 150
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_A = "gen_a.ckpt"
CHECKPOINT_GEN_B = "gen_b.ckpt"
CHECKPOINT_CRITIC_A = "disc_a.ckpt"
CHECKPOINT_CRITIC_B = "disc_b.ckpt"
WHICH_MODEL = "32epoch_1.0_identityloss/"
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