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
        checkpoint = torch.load("model/trained/gen_b.ckpt", map_location=config.DEVICE)
        person2genshin.load_state_dict(checkpoint["state_dict"])
        print("person2genshin load success")
        genshin2person = generator.Generator(3).to(config.DEVICE)
        checkpoint = torch.load("model/trained/gen_a.ckpt", map_location=config.DEVICE)
        genshin2person.load_state_dict(checkpoint["state_dict"])
        print("genshin2person load success")

        img = np.array(Image.open("output_img/test.jpg").convert("RGB"))
        augmentations = config.transforms(image=img)
        img = augmentations["image"].to(config.DEVICE)
        print(img.shape)
        fake_img = genshin2person(img)
        save_image(fake_img * 0.5 + 0.5, f"output_img/test_out.jpg")
        print("save success")

