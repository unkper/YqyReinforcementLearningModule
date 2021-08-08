import numpy as np
from social_force.vector3 import Vector3

class Steering():
    def __init__(self):
        self.weight = 1

    def reset(self):
        pass

    def force(self)->Vector3:
        pass