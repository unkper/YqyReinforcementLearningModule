from Box2D import b2ContactListener, b2Contact
from env.objects import BoxWall, Exit, Person
from env.utils.misc import ObjectType

class MyContactListener(b2ContactListener):
    def __init__(self, model):
        super(MyContactListener, self).__init__()
        self.model = model

    def BeginContact(self, contact:b2Contact):
        infoA, infoB = contact.fixtureA.userData, contact.fixtureB.userData
        if (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Exit) or (infoA.type == ObjectType.Exit and infoB.type == ObjectType.Agent):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            exit = infoA if infoA.type == ObjectType.Exit else infoB
            agent_model = self.model.find_person(agent.ID)
            #将agent的is_done置为true并删除其刚体
            agent_model.is_done = True
            print("One Agent{} has reached exit{}!!!".format(agent.ID, exit.ID))
        # print("Fixture A:{},Fixture B:{} begin contact!!!".format(contact.fixtureA.userData.ID,
        #                                                           contact.fixtureB.userData.ID))

    def EndContact(self, contact:b2Contact):
        pass
        # print("Fixture A:{},Fixture B:{} end contact!!!".format(contact.fixtureA.userData.ID,
        #                                                           contact.fixtureB.userData.ID))