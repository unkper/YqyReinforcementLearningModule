import pyglet
from pyglet.gl import *

# 创建窗口
window = pyglet.window.Window()

# 定义长方形的四个点坐标
vertices = [
    0, 0,
    window.width, 0,
    window.width, window.height,
    0, window.height
]

# 定义长方形的颜色，这里设为蓝色
colors = [0, 0, 1] * 4

@window.event
def on_draw():
    # 清除屏幕
    glClear(GL_COLOR_BUFFER_BIT)

    # 绘制长方形
    glLoadIdentity()
    glTranslatef(window.width // 2, window.height // 2, 0)
    glScalef(window.width // 2, window.height // 2, 1)
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, colors)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    pyglet.graphics.draw(4, GL_QUADS, ('v2f', vertices))

# 启动事件循环
pyglet.app.run()