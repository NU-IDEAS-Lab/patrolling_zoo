import networkx as nx
import numpy as np
from IPython.display import clear_output

from .algorithm import BaseAlgorithm


class GreedyIdleness(BaseAlgorithm):
    ''' This algorithm implements an idleness-based greedy heuristic for the patrolling problem. '''

    def save(self, *args, **kwargs):
        ''' Saves the model. '''
        pass


    def train(self, *args, seed=None, **kwargs):
        ''' Trains the model. '''
        pass

    
    def _reset(self):
        pass


    def evaluate(self, render=False, render_terminal=False, max_cycles=None, max_episodes=1, seed=None):
        ''' Evaluates the model. '''
        
        if max_cycles != None:
            self.env.max_cycles = max_cycles

        for episode in range(max_episodes):
            obs, info = self.env.reset(seed=seed)
            self._reset()

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
            
            if render_terminal:
                clear_output(wait=True)
                self.env.render()


    def generateActions(self, obs):
        ''' Generates a dictionary of actions. '''
        
        actions = {}

        for agent in self.env.agents:
            vertexStates = obs[agent]["vertex_state"]
            targetNode = max(vertexStates, key=vertexStates.get)
            actions[agent] = targetNode
        
        return actions


class GreedyDistance(GreedyIdleness):
    ''' This algorithm implements an distance-based greedy heuristic for the patrolling problem. '''

    def _reset(self):
        for agent in self.env.agents:
            agent.nodes = list(self.env.pg.graph.nodes)
            agent.nodes.remove(agent.startingNode)

    def generateActions(self, obs):
        ''' Generates a dictionary of actions. '''
        
        actions = {}

        # Check whether any agents have reached their target node.
        for agent in self.env.agents:
            vertexDistances = obs[agent]["vertex_distances"][agent]

            # Remove nodes which have already been visited.
            for node in agent.nodes:
                if agent.lastNode == node and agent.edge == None:
                    agent.nodes.remove(node)
            
            # If no nodes remain, reset the list of nodes.
            if len(agent.nodes) == 0:
                agent.nodes = list(self.env.pg.graph.nodes)
            
            # For any node which the agent has already visited, set its distance to infinity.
            for i in range(len(vertexDistances)):
                if i not in agent.nodes:
                    vertexDistances[i] = np.inf

            targetNode = np.argmin(vertexDistances)
            actions[agent] = targetNode
        
        return actions