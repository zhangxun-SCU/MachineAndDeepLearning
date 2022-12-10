import pygame
from pygame.sprite import Sprite
import random


class Barrier(Sprite):
    """障碍"""

    def __init__(self, ai_game):
        """初始化"""
        super().__init__()
        self.screen = ai_game.screen
        self.screen_rect = ai_game.screen.get_rect()
        # 设置障碍图像
        top_l = random.randint(100, 1100)
        self.barrier_top = pygame.Surface((80, top_l), flags=pygame.HWSURFACE)
        self.barrier_bottom = pygame.Surface((80, 1200 - top_l), flags=pygame.HWSURFACE)
        self.barrier_top.fill(color="pink")
        self.barrier_bottom.fill(color="pink")
        self.rect_top = self.barrier_top.get_rect()
        self.rect_bottom = self.barrier_bottom.get_rect()
        # 设置障碍的位置
        self.rect_top.top = self.screen_rect.top
        self.rect_bottom = self.screen_rect.bottom

    def blitme(self):
        self.screen.blit(self.barrier_top, self.rect_top)
        self.screen.blit(self.barrier_bottom, self.rect_bottom)

