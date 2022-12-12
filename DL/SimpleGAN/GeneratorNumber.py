import torch
import torchvision

from SimpleGAN import Generator

z_dim = 64
img_dim = 28 * 28 * 1

if __name__ == "__main__":
    with torch.no_grad():
        numberGenerator = Generator(z_dim, img_dim)
        numberGenerator.load_state_dict(torch.load("model/1000epoch/Generator.ckpt"))
        print("Load Success.")
        noise = torch.randn(1, z_dim)
        fake = numberGenerator(noise).reshape(-1, 1, 28, 28)
        img_grid = torchvision.utils.make_grid(fake, normalize=True)
        torchvision.utils.save_image(img_grid, r'imgs/test.png')



