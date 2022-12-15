import pygame
from pygame.sprite import Sprite
import random


class Barrier(Sprite):
    """障碍"""

    def __init__(self, ai_game, height, bottom=True):
        """初始化"""
        super().__init__()
        self.screen = ai_game.screen
        self.screen_rect = ai_game.screen.get_rect()
        # 设置障碍图像
        print(height)
        self.image = pygame.Surface((80, height), flags=pygame.HWSURFACE)

        self.image.fill(color="pink")

        self.rect = self.image.get_rect()
        # 设置障碍的位置
        if bottom:
            self.rect.bottom = self.screen_rect.bottom
        else:
            self.rect.top = self.screen_rect.top
        self.rect.right = self.screen_rect.right

        self.x = float(self.rect.x)

    def update(self):
        self.x -= 0.3
        self.rect.x = self.x

    def blitme(self):
        self.screen.blit(self.image, self.rect)

