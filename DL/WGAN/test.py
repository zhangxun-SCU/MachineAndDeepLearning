import os
import torch
import torchvision

from WGAN import Generator

z_dim = 64 * 64 * 3


if __name__ == '__main__':
    with torch.no_grad():
        gen = Generator(z_dim, 3, 64)
        gen.load_state_dict(torch.load("model/trained/Generator.151.ckpt", map_location='cuda:0'))
        print("Load Success.")
        # noise = torch.zeros((32, z_dim, 1, 1))
        noise = torch.ones((32, z_dim, 1, 1))
        noise += 1
        # noise = torch.randn(32, z_dim, 1, 1)
        fake = gen(noise)
        img_grid = torchvision.utils.make_grid(fake, normalize=True)
        torchvision.utils.save_image(img_grid, r'output_img/test1.png')
