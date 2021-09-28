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
            if agent.model.is_done == True:return
            #将agent的is_done置为true并删除其刚体
            agent.model.is_done = True
            # print("One Agent{} has reached exit{}!!!".format(agent.ID, exit.ID))
        elif (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Agent):
            self.col_with_agent += 1
            infoA.model.collide_with_agent = True
            infoB.model.collide_with_agent = True
        elif (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Wall) or (infoA.type == ObjectType.Wall and infoB.type == ObjectType.Agent):
            self.col_with_wall += 1
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            agent.model.collide_with_wall = True

        # print("Fixture A:{},Fixture B:{} begin contact!!!".format(contact.fixtureA.userData.ID,
        #                                                           contact.fixtureB.userData.ID))

    def EndContact(self, contact:b2Contact):
        infoA, infoB = contact.fixtureA.userData, contact.fixtureB.userData
        if (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Agent):
            infoA.model.collide_with_agent = False
            infoB.model.collide_with_agent = False
        elif (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Wall) or (infoA.type == ObjectType.Wall and infoB.type == ObjectType.Agent):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            agent.model.collide_with_wall = False
        # print("Fixture A:{},Fixture B:{} end contact!!!".format(contact.fixtureA.userData.ID,
        #                                                           contact.fixtureB.userData.ID))

