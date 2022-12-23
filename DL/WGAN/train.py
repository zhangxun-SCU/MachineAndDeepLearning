import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from WGAN64 import Critical, Generator, initialize_weights
from dataset import Dataset
from utils import gradient_penalty

# 设置超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-4
batch_size = 64
image_size = 64
channels_img = 3
z_dim = 100
num_epochs = 150
features_disc = 64
features_gen = 64
critic_iterations = 5
# weight_clip = 0.01
lambda_gp = 10

transforms = transforms.Compose(
    [transforms.Resize([image_size, image_size]),
     transforms.ToTensor(),
     transforms.Normalize(
         [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
     )]
)

# disc_path = r'model/Discriminator.ckpt'
# gen_path = r'model/Generator.ckpt'

dataset = Dataset("data/ganyu-final", transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


gen = Generator(z_dim, channels_img, features_gen).to(device)
critic = Critical(channels_img, features_disc).to(device)
initialize_weights(gen)
initialize_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=learning_rate, betas=(0.0, 0.9))

fixed_noise = torch.randn(4, z_dim, 1, 1).to(device)
step = 0

gen.train()
critic.train()

# # if os.path.exists(disc_path):
# disc.load_state_dict(torch.load("model/Discriminator51.ckpt"))
# print("Discriminator loaded")
# # if os.path.exists(gen_path):
# gen.load_state_dict(torch.load("model/Generator.51.ckpt"))
# print("Generator loaded")

for epoch in range(num_epochs):
    for batch_idx, real in enumerate(dataloader):
        real = real.to(device)
        # 训练disc max log(D(x)) + log(1 - D(G(z)))
        for _ in range(critic_iterations):
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)

            gp = gradient_penalty(critic, real, fake, device)
            loss_critic = (-(torch.mean(critic_real) - torch.mean(critic_fake)) + lambda_gp*gp)
            critic.zero_grad()
            loss_critic.backward()
            opt_critic.step()

        # print(fake.shape)

        # 训练生成器 min log(1 - D(G(z))) <--> max log(D(G(z)))
        noise = torch.randn((batch_size, z_dim, 1, 1)).to(device)
        fake = gen(noise)
        output = critic(fake).view(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(f"Epoch [{epoch}/{num_epochs}] Loss D:{loss_critic:.4f} Loss G:{loss_gen:.4f}")
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:4], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:4], normalize=True)
                torchvision.utils.save_image(img_grid_real, r'output_img/real/%d.png' % step)
                torchvision.utils.save_image(img_grid_fake, r'output_img/fake/%d.png' % step)

                step = step + 1

    torch.save(critic.state_dict(), f"model/Discriminator.ckpt")
    torch.save(gen.state_dict(), f"model/Generator.ckpt")
    print("model saved")
