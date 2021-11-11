from Box2D import b2ContactListener, b2Contact
from ped_env.utils.misc import ObjectType

class MyContactListener(b2ContactListener):
    def __init__(self, world):
        super(MyContactListener, self).__init__()
        self.world = world
        self.col_with_agent = 0
        self.col_with_wall = 0

    def BeginContact(self, contact:b2Contact):
        infoA, infoB = contact.fixtureA.userData, contact.fixtureB.userData
        if (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Exit) or (infoA.type == ObjectType.Exit and infoB.type == ObjectType.Agent):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            exit = infoA if infoA.type == ObjectType.Exit else infoB
            if agent.env.is_done == True:return
            #只有exit_type匹配时，才将agent的is_done置为true并删除其刚体
            if agent.env.exit_type == exit.env.exit_type:
                agent.env.is_done = True
                # print("One Agent{} has reached exit{}!!!".format(agent.id, exit.id))
        elif (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Agent):
            self.col_with_agent += 1
            infoA.env.collide_with_agent = True
            infoB.env.collide_with_agent = True
        elif (infoA.type == ObjectType.Agent and infoB.type in (ObjectType.Wall, ObjectType.Obstacle)) \
                or (infoA.type in (ObjectType.Wall, ObjectType.Obstacle) and infoB.type == ObjectType.Agent):
            self.col_with_wall += 1
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            agent.env.collide_with_wall = True

    def EndContact(self, contact:b2Contact):
        infoA, infoB = contact.fixtureA.userData, contact.fixtureB.userData
        if (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Agent):
            infoA.env.collide_with_agent = False
            infoB.env.collide_with_agent = False
        elif (infoA.type == ObjectType.Agent and infoB.type in (ObjectType.Wall, ObjectType.Obstacle)) \
                or (infoA.type in (ObjectType.Wall, ObjectType.Obstacle) and infoB.type == ObjectType.Agent):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            agent.env.collide_with_wall = False
        # print("Fixture A:{},Fixture B:{} end contact!!!".format(contact.fixtureA.userData.ID,
        #                                                           contact.fixtureB.userData.ID))

