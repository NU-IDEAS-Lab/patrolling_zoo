import math
import random 

class Comm_model():
    def __init__(self, p=0.7, alpha=0.3, beta=0.7, gamma=3, pt = 30.0, ps=-65.0):
        self.p = p
        self.alpha =alpha
        self.beta = beta
        self.gamma = gamma
        self.pt = pt
        self.ps = ps


    def bernouli_model(self):
        if random.random()< self.p:
            return True
        else:
            return False
        
    def Gil_el_model(self,agent):
        if agent.currentState:
            change_state = (random.random()<self.alpha)
            if change_state:
                agent.currentState = not agent.currentState
            if agent.currentState:
                return True
            else:
                return False
            
        else:
            change_state = (random.random()<self.beta)
            if change_state:
                agent.currentState = not agent.currentState
            if agent.currentState:
                return True
            else:
                return False
            

    def path_loss_model(self,agent_one, agent_two):
        x = abs(agent_one.position[0]-agent_two.position[0])
        y = abs(agent_one.position[0]-agent_two.position[0])
        distance = abs(math.sqrt(x*x + y*y))
        if distance == 0:
            return True
        p_receive = self.pt - 10*self.gamma*math.log10(distance)
        return p_receive>self.ps
