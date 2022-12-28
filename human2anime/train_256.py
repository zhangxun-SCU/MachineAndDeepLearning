import torch
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
from dataset import MyDataSet


def train_fn(disc_a, disc_b, gen_a, gen_b, dataloader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    loop = tqdm(dataloader, leave=True)

    for idx, (a, b) in enumerate(loop):
        a = a.to(config.DEVICE)
        b = b.to(config.DEVICE)

        # Train Discriminators a,b
        with torch.cuda.amp.autocast():
            fake_b = gen_b(a)
            disc_b_real = disc_b(b)
            disc_b_fake = disc_b(fake_b.detach())
            disc_b_real_loss = mse(disc_b_real, torch.ones_like(disc_b_real))
            disc_b_fake_loss = mse(disc_b_fake, torch.zeros_like(disc_b_fake))
            disc_b_loss = disc_b_fake_loss + disc_b_real_loss

            fake_a = gen_a(b)
            disc_a_real = disc_a(a)
            disc_a_fake = disc_a(fake_a.detach())
            disc_a_real_loss = mse(disc_a_real, torch.ones_like(disc_a_real))
            disc_a_fake_loss = mse(disc_a_fake, torch.zeros_like(disc_a_fake))
            disc_a_loss = disc_a_fake_loss + disc_a_real_loss

            # 将两种损失组合
            disc_loss = (disc_a_loss + disc_b_loss) / 2

        opt_disc.zero_grad()
        d_scaler.scale(disc_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # train Generator
        with torch.cuda.amp.autocast():
            disc_a_fake = disc_a(fake_a)
            disc_b_fake = disc_b(fake_b)
            gen_a_loss = mse(disc_a_fake, torch.ones_like(disc_a_fake))
            gen_b_loss = mse(disc_b_fake, torch.ones_like(disc_b_fake))

            # cycle loss
            cycle_b = gen_b(fake_a)
            cycle_a = gen_a(fake_b)
            cycle_b_loss = l1(b, cycle_b)
            cycle_a_loss = l1(a, cycle_a)

            # identity loss
            identity_b = gen_b(b)
            identity_a = gen_a(a)
            identity_b_loss = l1(b, identity_b)
            identity_a_loss = l1(a, identity_a)

            gen_loss = (
                gen_a_loss +
                gen_b_loss +
                cycle_a_loss * config.LAMBDA_CYCLE +
                cycle_b_loss * config.LAMBDA_CYCLE +
                identity_a_loss * config.LAMBDA_IDENTITY +
                identity_b_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 1000 == 0:
            save_image(fake_a*0.5+0.5, f"output_img/fake_anime/fake_anime_{idx}.png")
            save_image(fake_b*0.5+0.5, f"output_img/fake_human/fake_human_{idx}.png")


def main():
    disc_a = Discriminator(in_channels=3).to(config.DEVICE)
    disc_b = Discriminator(in_channels=3).to(config.DEVICE)

    gen_a = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_b = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

    opt_disc = optim.Adam(
        list(disc_a.parameters()) + list(disc_b.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_b.parameters()) + list(gen_a.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    MSE = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_GEN_A, gen_a, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_GEN_B, gen_b, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_CRITIC_A, disc_a, opt_disc, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_CRITIC_B, disc_b, opt_disc, config.LEARNING_RATE,
        )

    dataset = MyDataSet(
        root_a="dataset/anime",
        root_b="dataset/human",
        transform=config.transforms,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        print(f" epoch/epochs: {epoch}/{config.NUM_EPOCHS}")
        train_fn(
            disc_a=disc_a,
            disc_b=disc_b,
            gen_a=gen_a,
            gen_b=gen_b,
            dataloader=dataloader,
            opt_disc=opt_disc,
            opt_gen=opt_gen,
            l1=L1,
            mse=MSE,
            d_scaler=d_scaler,
            g_scaler=g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_a, opt_gen,
                            filename=(config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_GEN_A))
            save_checkpoint(gen_b, opt_gen,
                            filename=(config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_GEN_B))
            save_checkpoint(disc_a, opt_disc,
                            filename=(config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_CRITIC_A))
            save_checkpoint(disc_b, opt_disc,
                            filename=(config.MODEL_DIR + config.WHICH_MODEL + config.CHECKPOINT_CRITIC_B))


if __name__ == "__main__":
    main()
