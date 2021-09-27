import pyglet
import socket
import Box2D as b2d

from ped_env.utils.viewer import MyDrawer
from pygame import Color

class SimuationPedEnvironment():
    '''
    作为社会力和强化学习算法的观察数据提供者和动作转发者，该对象能同时维护多个Env类，
    在每一次step时，都会将所需信息提供给相应Env然后获得对应的力，其将所有力相加成一个
    合力后返回给前端（通常是Unity,Unreal等游戏框架)，数据通信暂定为：
    Tcp通信，绑定端口号12578，每个包包含消息头（调用消息类型，总消息长度）
    1.reset方法，提示服务端重置环境并返回初始观察数据
    2.close方法，提示服务端关闭环境
    3.seed方法，包含种子数据用于随机
    4.step方法，包含动作数据（给智能体应该施加的力向量），返回（观察信息，*奖励值，is_done,info）
    5.get_position方法，返回当前所有智能体的位置信息
    该类应该完成的：管理并调用各个智能体的控制Env，得到应该施加的力并将它们合成
    '''
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        self.socket.connect(('127.0.0.1',12578))

    def step(self, action):
        pass

    def close(self):
        pass

    def reset(self):
        pass

    def seed(self, seed=None):
        pass

    def render(self, mode='human'):
        pass
