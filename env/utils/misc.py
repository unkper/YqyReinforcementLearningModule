import enum

class ObjectType(enum.Enum):
    Agent = 1
    Wall = 2
    Exit = 3

class FixtureInfo():
    def __init__(self, id:int, model:object, type:ObjectType):
        self.ID = id
        self.model = model
        self.type = type

    def __str__(self):
        return str(self.type) + str(self.ID)
