import pygame
from pygame.sprite import Sprite


class Bird(Sprite):
    """操控的小鸟"""

    def __init__(self, ai_game):
        """初始化"""
        super().__init__()
        self.screen = ai_game.screen
        self.screen_rect = ai_game.screen.get_rect()
        # 加载小鸟图像资源并设置其位置
        self.bird_img = pygame.image.load("images/pl.bmp")
        self.rect = self.bird_img.get_rect()
        self.rect.midleft = self.screen_rect.midleft
        self.rect.x = 200

        # 小鸟是否飞（空格）
        self.fly_up = False

        # 为了使游戏减慢速度，以小数心事存储位置信息
        self.y = float(self.rect.y)

    def update(self):
        if self.rect.bottom < self.screen_rect.bottom:
            self.y += 0.4
        if self.fly_up:
            self.y -= 0.8

        self.rect.y = self.y

    def blitme(self):
        """绘制小鸟"""
        self.screen.blit(self.bird_img, self.rect)