import pygame
import math

# 初始化 Pygame
pygame.init()

# 设置窗口大小和标题
size = (500, 500)
screen = pygame.display.set_mode(size)
pygame.display.set_caption("行走的人物和方向")

# 设置人物圆的位置、半径和速度
circle_pos = [250, 250]
circle_radius = 50
circle_speed = 5

# 设置方向三角形的长度和角度
triangle_length = 30
triangle_angle = 45

# 定义函数，将角度转换为弧度
def deg_to_rad(degrees):
    return degrees * 3.1415 / 180

# 定义函数，计算三角形的顶点坐标
def calc_triangle_points(pos, length, angle):
    # 将角度转换为弧度
    angle_rad = deg_to_rad(angle)

    # 计算三角形的顶点坐标
    x1 = pos[0] + length * math.cos(angle_rad)
    y1 = pos[1] - length * math.sin(angle_rad)

    x2 = pos[0] + length * math.cos(angle_rad + deg_to_rad(120))
    y2 = pos[1] - length * math.sin(angle_rad + deg_to_rad(120))

    x3 = pos[0] + length * math.cos(angle_rad + deg_to_rad(240))
    y3 = pos[1] - length * math.sin(angle_rad + deg_to_rad(240))

    return [(x1, y1), (x2, y2), (x3, y3)]

# 循环运行，直到用户关闭窗口
done = False
while not done:
    # 处理事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        # 检测按键事件
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                # 左移
                circle_pos[0] -= circle_speed
                triangle_angle = 180
            elif event.key == pygame.K_RIGHT:
                # 右移
                circle_pos[0] += circle_speed
                triangle_angle = 0
            elif event.key == pygame.K_UP:
                # 上移
                circle_pos[1] -= circle_speed
                triangle_angle = 90
            elif event.key == pygame.K_DOWN:
                # 下移
                circle_pos[1] += circle_speed
                triangle_angle = 270

    # 填充背景色
    screen.fill((255, 255, 255))

    # 绘制人物圆
    pygame.draw.circle(screen, (255, 0, 0), circle_pos, circle_radius)

    # 计算三角形顶点坐标
    triangle_points = calc_triangle_points(circle_pos, triangle_length, triangle_angle)

    # 绘制三角形
    pygame.draw.polygon(screen, (0, 0, 255), triangle_points)

    # 更新屏幕显示
    pygame.display.flip()

# 退出 Pygame
