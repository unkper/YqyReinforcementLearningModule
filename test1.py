import pyglet
import numpy as np

# 创建一个窗口
window = pyglet.window.Window()

# 渲染五角星
def draw_star(x, y, size, color):
    half_size = size / 2
    vertices = [
        x, y + half_size,
        x + half_size, y + size,
        x + size, y + half_size,
        x + half_size, y,
        x, y + half_size
    ]
    colors = list(color) * 5
    pyglet.graphics.draw(5, pyglet.gl.GL_LINE_STRIP, ('v2f', vertices), ('c3B', colors))

# 在窗口中渲染图像
@window.event
def on_draw():
    window.clear()
    draw_star(window.width / 2, window.height / 2, 150, (255, 255, 255, 255))

# 在窗口中运行pyglet应用程序
pyglet.app.run()

# 获取渲染后的颜色缓冲区
color_buffer = pyglet.image.get_buffer_manager().get_color_buffer()

# 将颜色缓冲区转换为numpy数组
data = np.frombuffer(color_buffer.get_image_data().get_data('RGB', color_buffer.pitch), dtype=np.uint8)
data = data.reshape((color_buffer.height, color_buffer.width, 3))