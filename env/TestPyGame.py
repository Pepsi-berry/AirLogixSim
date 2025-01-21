import pygame


class TestPyGame:
    @staticmethod
    def render():
        # 初始化 Pygame
        pygame.init()

        # 创建窗口
        screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption("Reinforcement Learning Environment")

        # 定义颜色
        WHITE = (255, 255, 255)
        RED = (255, 0, 0)

        rect = (100, 150, 50, 50)

        # 游戏主循环
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_UP:
                        rect = (rect[0], rect[1] - 10, rect[2], rect[3])
                    if event.key == pygame.K_DOWN:
                        rect = (rect[0], rect[1] + 10, rect[2], rect[3])
                    if event.key == pygame.K_LEFT:
                        rect = (rect[0] - 10, rect[1], rect[2], rect[3])
                    if event.key == pygame.K_RIGHT:
                        rect = (rect[0] + 10, rect[1], rect[2], rect[3])

            # 绘制背景
            screen.fill(WHITE)

            # 绘制示例矩形
            pygame.draw.rect(screen, RED, rect)

            # 更新显示
            pygame.display.flip()

        # 退出 Pygame
        pygame.quit()
