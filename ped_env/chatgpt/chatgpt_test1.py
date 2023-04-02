import pygame
from Box2D import *

# 初始化 Pygame 和 Box2D
pygame.init()
world = b2World(gravity=(0, -10))

# 创建一个刚体
body_def = b2BodyDef()
body_def.position = (0, 0)
body = world.CreateDynamicBody()

# 创建一个圆形夹具
fixture_def = b2FixtureDef()
fixture_def.shape = b2CircleShape(radius=1)
body.CreateFixture(fixture_def)

# 设置刚体的朝向
body.angle = 0.5  # 弧度制

# 创建一个 Pygame 窗口
screen = pygame.display.set_mode((640, 480))

# 设置颜色
white = (255, 255, 255)
red = (255, 0, 0)

# 运行主循环
running = True
while running:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 清空屏幕
    screen.fill(white)

    # 获取小球的位置和朝向
    pos = body.position
    angle = body.angle

    # 绘制小球
    pygame.draw.circle(screen, red, (int(pos.x), int(pos.y)), 20)

    # 计算朝向指示器的三个顶点
    direction = b2Rot(angle).GetUnitVector()
    p1 = pos + 20 * direction
    p2 = pos + 10 * b2Rot(angle + 135 / 180 * b2_pi).GetUnitVector()
    p3 = pos + 10 * b2Rot(angle - 135 / 180 * b2_pi).GetUnitVector()

    # 绘制朝向指示器
    pygame.draw.polygon(screen, red, [p1, p2, p3])

    # 更新屏幕
    pygame.display.flip()

# 关闭 Pygame 和 Box2D
pygame.quit()
