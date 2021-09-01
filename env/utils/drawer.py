import sys

import Box2D as b2d
import pygame


class MyDrawer():
    def __init__(self):
        super(MyDrawer, self).__init__()
        pygame.init() #init pygame module
        self.surface = pygame.display.set_mode((400, 400), flags=pygame.RESIZABLE)
        self.physics_scale = 1 / 30
        self.ref_res = (400, 400)

        self.render_scale = (self.surface.get_size()[0] / self.ref_res[0],
                             self.surface.get_size()[1] / self.ref_res[1])


    def DrawCircle(self, center, radius, color, width = 1):
        pygame.draw.circle(
            self.surface,
            color,
            (int(center[0] / self.physics_scale * self.render_scale[0]),
             int(center[1] / self.physics_scale * self.render_scale[1])),
            int(radius / self.physics_scale * self.render_scale[0]),
            width=width)

    def DrawSolidCircle(self, center, radius, axis, color):
        self.DrawCircle(center, radius, color, width=0)

    def DrawPolygon(self, vertices, vertexCount, color, width=1):
        if vertexCount != 4:raise Exception("Current Only Support Draw Square!!!")
        pygame.draw.polygon(
            self.surface, color,
            ((vertices[0][0] / self.physics_scale * self.render_scale[0],
              vertices[0][1] / self.physics_scale * self.render_scale[1]),
             (vertices[1][0] / self.physics_scale * self.render_scale[0],
              vertices[1][1] / self.physics_scale * self.render_scale[1]),
             (vertices[2][0] / self.physics_scale * self.render_scale[0],
              vertices[2][1] / self.physics_scale * self.render_scale[1]),
             (vertices[3][0] / self.physics_scale * self.render_scale[0],
              vertices[3][1] / self.physics_scale * self.render_scale[1])),
            width=width)

    def DrawSolidPolygon(self, vertices, vertexCount, color):
        self.DrawPolygon(vertices, vertexCount, color, width=0)

    def DrawLine(self, points, color, width=1):
        pygame.draw.lines(
            self.surface,color,
            True,points,width
        )

    def update(self):
        for event in pygame.event.get():  # 循环获取事件
            if event.type == pygame.QUIT:  # 若检测到事件类型为退出，则退出系统
                pygame.quit()
                sys.exit()
        pygame.display.update()  # 更新屏幕内容