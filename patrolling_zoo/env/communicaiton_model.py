import math
import random 

class Comm_model():
    def bernouli_model(self, p):
        if random.random()< p:
            return True
        else:
            return False
        
    def Gil_el_model(self,agent, alpha,beta):
        if agent.currentState:
            change_state = (random.random()<alpha)
            if change_state:
                agent.currentState = not agent.currentState
            if agent.currentState:
                return True
            else:
                return False
            
        else:
            change_state = (random.random()<beta)
            if change_state:
                agent.currentState = not agent.currentState
            if agent.currentState:
                return True
            else:
                return False
            

    def path_loss_model(self,agent_one, agent_two, gamma, pt, ps):
        x = abs(agent_one.position[0]-agent_two.position[0])
        y = abs(agent_one.position[0]-agent_two.position[0])
        distance = abs(math.sqrt(x*x + y*y))
        p_receive = pt - 10*gamma*math.log10(distance)
        return p_receive>ps
