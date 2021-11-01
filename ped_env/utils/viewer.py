import sys

import Box2D as b2d

import pyglet

class PedsMoveEnvViewer(pyglet.window.Window):
    def __init__(self, model):
        super().__init__(width=500,
                         height=500,
                         caption="行人行走模拟环境",
                         config=pyglet.gl.Config(double_buffer=True),
                         visible=True)

        # This call schedules the `update()` method to be called
        # TICKS_PER_SEC. This is the main game event loop.
        # pyglet.clock.schedule_interval(self.update, 1.0 / TICKS_PER_SEC)
        # 窗口背景颜色
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.model = model

    def render(self):
        self.model.setup_graphics()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.model.batch.draw()