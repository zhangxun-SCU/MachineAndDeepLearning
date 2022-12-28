from PIL import Image
import cv2
import os

# for (int i = 1; i <= 10; i++)
dir_list = os.listdir("dataset/img_align_celeba/")
for img_name in dir_list:
    img = Image.open(f"dataset/img_align_celeba/{img_name}")

    cropped = img.crop((0, 20, 178, 198))

    cropped.save(f"dataset/human/{img_name}")
