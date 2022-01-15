from Box2D import b2ContactListener, b2Contact
from ped_env.utils.misc import ObjectType

class MyContactListener(b2ContactListener):
    def __init__(self, env):
        super(MyContactListener, self).__init__()
        self.env = env

    def BeginContact(self, contact:b2Contact):
        infoA, infoB = contact.fixtureA.userData, contact.fixtureB.userData
        if (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Exit) or (infoA.type == ObjectType.Exit and infoB.type == ObjectType.Agent):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            exit = infoA if infoA.type == ObjectType.Exit else infoB
            if agent.model.is_done == True:
                return
            #只有exit_type匹配时，才将agent的is_done置为true，删除刚体的流程放在循环中进行
            if agent.model.exit_type == exit.model.exit_type:
                agent.model.is_done = True
            # print("One Agent{} has reached exit{}!!!".format(agent.id, exit.id))
        elif (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Agent):
            if infoB.model in infoA.model.group:
                return
            self.env.col_with_agent += 1
            #互相添加彼此
            infoA.model.collide_agents[infoB.id] = infoB.model
            infoB.model.collide_agents[infoA.id] = infoA.model

        elif (infoA.type == ObjectType.Sensor and infoB.type == ObjectType.Agent) or (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Sensor):
            infoA.model.detected_agents[infoB.id] = infoB.model
            infoB.model.detected_agents[infoA.id] = infoA.model

        elif (infoA.type == ObjectType.Agent and infoB.type in (ObjectType.Wall, ObjectType.Obstacle)) \
                or (infoA.type in (ObjectType.Wall, ObjectType.Obstacle) and infoB.type == ObjectType.Agent):
            self.env.col_with_wall += 1
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            obs = infoA if infoA.type in (ObjectType.Wall, ObjectType.Obstacle) else infoB
            agent.model.collide_obstacles[obs.id] = obs.model

        elif (infoA.type == ObjectType.Sensor and infoB.type in (ObjectType.Wall, ObjectType.Obstacle)) \
                or (infoA.type in (ObjectType.Wall, ObjectType.Obstacle) and infoB.type == ObjectType.Sensor):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            obs = infoA if infoA.type in (ObjectType.Wall, ObjectType.Obstacle) else infoB
            agent.model.detected_obstacles[obs.id] = obs.model
        else:
            pass
            #print("出现未知类型的碰撞!{}-{}".format(infoA.type, infoB.type))

    def EndContact(self, contact:b2Contact):
        infoA, infoB = contact.fixtureA.userData, contact.fixtureB.userData
        if (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Agent):
            if infoB.model in infoA.model.group:
                return
            if infoB.id in infoA.model.collide_agents.keys():infoA.model.collide_agents.pop(infoB.id)
            if infoA.id in infoB.model.collide_agents.keys():infoB.model.collide_agents.pop(infoA.id)

        elif (infoA.type == ObjectType.Sensor and infoB.type == ObjectType.Agent) or (infoA.type == ObjectType.Agent and infoB.type == ObjectType.Sensor):
            if infoB.id in infoA.model.detected_agents.keys():infoA.model.detected_agents.pop(infoB.id)
            if infoA.id in infoB.model.detected_agents.keys():infoB.model.detected_agents.pop(infoA.id)

        elif (infoA.type == ObjectType.Agent and infoB.type in (ObjectType.Wall, ObjectType.Obstacle)) \
                or (infoA.type in (ObjectType.Wall, ObjectType.Obstacle) and infoB.type == ObjectType.Agent):

            agent = infoA if infoA.type == ObjectType.Agent else infoB
            obs = infoA if infoA.type in (ObjectType.Wall, ObjectType.Obstacle) else infoB
            if obs.id in agent.model.collide_obstacles.keys():agent.model.collide_obstacles.pop(obs.id)

        elif (infoA.type == ObjectType.Sensor and infoB.type in (ObjectType.Wall, ObjectType.Obstacle)) \
                or (infoA.type in (ObjectType.Wall, ObjectType.Obstacle) and infoB.type == ObjectType.Sensor):
            agent = infoA if infoA.type == ObjectType.Agent else infoB
            obs = infoA if infoA.type in (ObjectType.Wall, ObjectType.Obstacle) else infoB
            if obs.id in agent.model.detected_obstacles.keys():agent.model.detected_obstacles.pop(obs.id)
        else:
            pass
            #print("出现未知类型的碰撞!{}-{}".format(infoA.type, infoB.type))
