import logging
import time

import pyglet

from ped_env.settings import TICKS_PER_SEC


class PedsMoveEnvRender():
    def __init__(self, env, render_ratio=1):
        self.width = int(500 * render_ratio)
        self.height = int(500 * render_ratio)
        self.env = env
        logging.warning(u"创建用于渲染输出状态的渲染器!")

    def render(self):
        self.env._render()
        framebuffer = pyglet.image.get_buffer_manager().get_color_buffer()
        pyglet.gl.glClearColor(0, 0, 0, 1)
        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)
        self.env.batch.draw()
        pyglet.gl.glClearColor(1, 1, 1, 1)
        pyglet.gl.glClear(pyglet.gl.GL_COLOR_BUFFER_BIT)


class PedsMoveEnvViewer(pyglet.window.Window):
    def __init__(self, env, visible=True, render_ratio=1):
        width = int(500 * render_ratio)
        height = int(500 * render_ratio)
        super().__init__(width=width,
                         height=height,
                         resizable=True,
                         caption="行人行走模拟环境",
                         config=pyglet.gl.Config(double_buffer=True),
                         visible=visible)

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        # pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)
        # 窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.env = env
        self.frame_time = time.time()
        self.cor_frame_time = self.env.frame_skipping * 1 / TICKS_PER_SEC
        self.pressed = False
        self.render_ratio = render_ratio

    def get_render_scale(self):
        """
        得到缩放比例，500*500的窗口大小
        :return:
        """
        size = self.env.terrain.map.shape[0]
        return 500 * self.render_ratio / size, 500 * self.render_ratio / size

    def render(self):
        self.env._render()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE and not self.pressed:
            self.pressed = True

    def on_draw(self):
        self.clear()
        if not self.visible:
            self.env.batch.draw()
        else:
            now_time = time.time()
            dt = now_time - self.frame_time
            self.frame_time = now_time
            self.env.batch.draw()
            time_in_env = self.env.step_in_env * 1 / 50
            if self.cor_frame_time < dt:
                self.set_caption("行人行走模拟环境,当前时间:{}".format(time_in_env) + "@@以更慢的速度渲染!")
            else:
                self.set_caption("行人行走模拟环境,当前时间:{}".format(time_in_env))
                time.sleep(self.cor_frame_time - dt)


class PedsMoveEnvViewerWithBuffer(pyglet.window.Window):
    def __init__(self, model):
        super().__init__(width=500,
                         height=500,
                         caption="行人行走模拟环境",
                         config=pyglet.gl.Config(double_buffer=True),
                         visible=True)
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.model = model
        self.buffers = []

    def render(self):
        self.model._render()
        self.buffers.append(self.model.batch)
