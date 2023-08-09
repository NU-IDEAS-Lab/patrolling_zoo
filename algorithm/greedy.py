import networkx as nx
from IPython.display import clear_output

from .algorithm import BaseAlgorithm


class Greedy(BaseAlgorithm):
    ''' This algorithm implements an idleness-based greedy heuristic for the patrolling problem. '''

    def save(self, *args, **kwargs):
        ''' Saves the model. '''
        pass


    def train(self, *args, seed=None, **kwargs):
        ''' Trains the model. '''
        pass


    def evaluate(self, render=False, max_cycles=None, max_episodes=1, seed=None):
        ''' Evaluates the model. '''
        
        if max_cycles != None:
            self.env.max_cycles = max_cycles

        for episode in range(max_episodes):
            obs, info = self.env.reset(seed=seed)

            if render:
                clear_output(wait=True)
                self.env.render()
            
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions = self.generateActions(obs)
                obs, rewards, terms, truncs, infos = self.env.step(actions)

                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                if render:
                    clear_output(wait=True)
                    self.env.render()


    def generateActions(self, obs):
        ''' Generates a dictionary of actions. '''
        
        actions = {}

        # Check whether any agents have reached their target node.
        for agent in self.env.agents:
            vertexStates = obs[agent]["vertex_state"]
            targetNode = max(vertexStates, key=vertexStates.get)
            actions[agent] = targetNode
        
        return actions
