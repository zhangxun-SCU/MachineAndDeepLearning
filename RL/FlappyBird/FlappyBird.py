import sys
import pygame
from Bird import Bird
from Barrier import Barrier
import random


class FlappyBird:
    """管理游戏资源"""

    def __init__(self):
        pygame.init()
        # 设置主窗口
        self.screen = pygame.display.set_mode((1200, 800))
        # 设置标题
        pygame.display.set_caption("FlappyBird")

        # 创建一只鸟
        self.bird = Bird(self)
        self.barriers = pygame.sprite.Group()

        # 自定义一个每秒的事件
        self.MYEVENT01 = pygame.USEREVENT + 1
        pygame.time.set_timer(self.MYEVENT01, 1400)
        clock = pygame.time.Clock()

    def run(self):
        self.create_barrier()
        while True:
            self.check_events()
            self.check_collision()
            self.update_screen()

    def update_screen(self):
        self.screen.fill((222, 222, 222))
        self.bird.update()
        self.bird.blitme()
        self.update_barrier()
        self.barriers.draw(self.screen)
        pygame.display.flip()

    def update_barrier(self):
        self.barriers.update()
        # 删除消失的障碍
        for barrier in self.barriers.copy():
            if barrier.rect.x <= -80:
                self.barriers.remove(barrier)

    def check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.bird.fly_up = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    self.bird.fly_up = False

            if event.type == self.MYEVENT01:
                self.create_barrier()

    def create_barrier(self):
        height = random.randint(100, 550)
        barrier_bottom = Barrier(self, height, True)
        barrier_top = Barrier(self, 550 - height, False)
        self.barriers.add(barrier_bottom)
        self.barriers.add(barrier_top)

    def check_collision(self):
        if pygame.sprite.spritecollideany(self.bird, self.barriers):
            exit()



if __name__ == "__main__":
    FlappyBird().run()
