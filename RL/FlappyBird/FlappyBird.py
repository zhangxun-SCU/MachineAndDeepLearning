import sys
import pygame
from Bird import Bird
from Barrier import Barrier


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

    def run(self):
        self.create_barrier()
        while True:
            self.check_events()

            self.update_screen()

    def update_screen(self):
        self.screen.fill((222, 222, 222))
        self.bird.update()
        self.bird.blitme()
        self.barriers.draw(self.screen)
        pygame.display.flip()

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

    def create_barrier(self):
        barrier = Barrier(self)
        self.barriers.add(barrier)


if __name__ == "__main__":
    FlappyBird().run()
