import torch
import config
import generator
import utils
from PIL import Image
import numpy as np
from torchvision.utils import save_image


if __name__ == "__main__":
    with torch.no_grad():
        person2genshin = generator.Generator(3).to(config.DEVICE)
        checkpoint = torch.load("model/500epoch_10_identityloss/gen_b.ckpt", map_location=config.DEVICE)
        person2genshin.load_state_dict(checkpoint["state_dict"])
        print("person2genshin load success")
        genshin2person = generator.Generator(3).to(config.DEVICE)
        checkpoint = torch.load("model/500epoch_10_identityloss/gen_a.ckpt", map_location=config.DEVICE)
        genshin2person.load_state_dict(checkpoint["state_dict"])
        print("genshin2person load success")
        img_a = np.array(Image.open("output_img/test_a.jpg").convert("RGB"))
        img_b = np.array(Image.open("output_img/test_b.jpg").convert("RGB"))
        augmentations = config.transforms(image=img_b, image0=img_a)
        img_a = augmentations["image0"].to(config.DEVICE)
        img_b = augmentations["image"].to(config.DEVICE)
        fake_a = person2genshin(img_b)
        save_image(fake_a * 0.5 + 0.5, f"output_img/test_b_out.jpg")
        fake_b = genshin2person(img_a)
        save_image(fake_b * 0.5 + 0.5, f"output_img/test_a_out.jpg")
        print("save success")

