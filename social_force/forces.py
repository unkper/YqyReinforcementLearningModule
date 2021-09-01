import numpy as np
from Box2D import b2Vec2

class Steering():
    def __init__(self):
        self.weight = 1

    def reset(self):
        pass

    def force(self)->b2Vec2:
        pass