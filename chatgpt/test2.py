import sys
from random import Random

import pygame

pygame.init()

# 创建一个窗口
screen = pygame.display.set_mode((640, 480))

# 创建一个 Surface 对象
surf = pygame.Surface((100, 100))



r = Random()
# 等待退出
while True:
    surf.fill((r.randint(0, 255), r.randint(0, 255), r.randint(0, 255)))  # 填充红色
    screen.fill((0, 0, 0))
    # 将 Surface 对象绘制到屏幕上
    screen.blit(surf, (r.randint(0, 300), r.randint(0, 200)))

    # 刷新屏幕
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()