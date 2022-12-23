
import torch
import torch.nn as nn


class Critical(nn.Module):
    """判别器"""
    def __init__(self, channels_img, features_d):
        super(Critical, self).__init__()
        self.disc = nn.Sequential(
            # 输入 N x channels_img x 64 x 64
            nn.Conv2d(
               channels_img, features_d, kernel_size=4, stride=2, padding=1  # N x feature_d x 32 x 32
            ),
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2, 1),  # 16 x 16
            self._block(features_d*2, features_d * 4, 4, 2, 1),  # 8 x 8
            self._block(features_d*4, features_d * 8, 4, 2, 1),  # 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),  # 1 x 1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """

        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :return:
        """
        return nn.Sequential(
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride, padding,
                      bias=False
                      ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        # 输入 N x z_dim x 1 x 1
        self.gen = nn.Sequential(
            self._block(z_dim, features_g * 16, 4, 2, 0),  # 4 x 4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32 x 32
            # self._block(features_g * 2, features_g * 1, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g*2,
                channels_img,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh()  # 归一化
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.gen(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W, = 8, 3, 64, 64
    z_dim = 128
    x = torch.randn((N, in_channels, H, W))
    disc = Critical(in_channels, 64)
    initialize_weights(disc)
    print(disc(x).shape)

    assert disc(x).shape == (N, 1, 1, 1)

    gen = Generator(z_dim, in_channels, 8)
    z = torch.randn((N, z_dim, 1, 1))
    print(gen(z).shape)
    assert  gen(z).shape == (N, in_channels, H, W)


if __name__ == "__main__":
    test()
