import enum

class ObjectType(enum.Enum):
    Agent = 1
    Wall = 2
    Obstacle = 3
    Exit = 4

class FixtureInfo():
    def __init__(self, id:int, model:object, type:ObjectType):
        self.id = id
        self.model = model
        self.type = type

    def __str__(self):
        return str(self.type) + str(self.id)


# counter = 0
# vecs = []
# for ped in self.peds:
#     if ped.body.linearVelocity.length > 1.6:
#         counter += 1
#         vecs.append(ped.body.linearVelocity.length)
# if counter > 0: print("目前有{}智能体超速，其速度为{}!".format(counter, vecs))

# for i,ped in enumerate(self.peds):
#     self.vec[i] += ped.body.linearVelocity.length
# if self.step_in_env % 200 == 0:
#     print("智能体平均速度为{}".format([x/200 for x in self.vec]))
#     self.vec = [0.0 for _ in range(len(self.peds))]