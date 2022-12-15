import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from DCGAN import Discriminator, Generator, initialize_weights
from dataset import Dataset

# 设置超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 2e-4
batch_size = 128
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 1000
features_disc = 64
features_gen = 64

transforms = transforms.Compose(
    [transforms.Resize(image_size),
     transforms.ToTensor(),
     transforms.Normalize(
         [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
     )]
)

disc_path = r'model/Discriminator.ckpt'
gen_path = r'model/Generator.ckpt'

dataset = Dataset("data", transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


gen = Generator(z_dim, channels_img, features_gen).to(device)
disc = Discriminator(channels_img, features_disc).to(device)

opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
opt_disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(0.5, 0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
step = 0
gen.train()
disc.train()

if os.path.exists(disc_path):
    disc.load_state_dict(torch.load(disc_path))
    print("Discriminator loaded")
if os.path.exists(gen_path):
    gen.load_state_dict(torch.load(gen_path))
    print("Generator loaded")

for epoch in range(num_epochs):
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device)
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)

        # 训练disc max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_fake + loss_disc_real) / 2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        # 训练生成器 min log(1 - D(G(z))) <--> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D:{loss_disc:.4f} Loss G:{loss_gen:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                torchvision.utils.save_image(img_grid_real, r'output_img/real/%d.png' % step)
                torchvision.utils.save_image(img_grid_fake, r'output_img/fake/%d.png' % step)

                step = step + 1
    if epoch % 50 == 0:
        torch.save(disc.state_dict(), f"model/Discriminator{epoch+1}.ckpt")
        torch.save(gen.state_dict(), gen_path)
        print("model saved")