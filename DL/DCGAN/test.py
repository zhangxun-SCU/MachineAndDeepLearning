import torch
import torchvision

from DCGAN import Generator

z_dim = 100

if __name__ == '__main__':
    with torch.no_grad():
        gen = Generator(z_dim, 3, 64)
        gen.load_state_dict(torch.load("model/generator.ckpt"))
        print("Load Success.")
        noise = torch.randn(1, 100, 1, 1)
        fake = gen(noise)
        img_grid = torchvision.utils.make_grid(fake, normalize=True)
        torchvision.utils.save_image(img_grid, r'output_img/test.png')
