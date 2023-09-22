import networkx as nx
import numpy as np
from IPython.display import clear_output
import random

from .algorithm import BaseAlgorithm


class RandomChoice(BaseAlgorithm):
    ''' This algorithm implements an idleness-based greedy heuristic for the patrolling problem. '''

    def save(self, *args, **kwargs):
        ''' Saves the model. '''
        pass


    def train(self, *args, seed=None, **kwargs):
        ''' Trains the model. '''
        pass

    
    def _reset(self):
        pass


    def evaluate(self, render=False, max_cycles=None, max_episodes=1, seed=None):
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

            self.prevActions = {a: None for a in self.env.agents}

            while not any(terms) and not any(truncs):
                actions = self.generateActions(obs)
                obs, rewards, terms, truncs, infos = self.env.step(actions)

                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                if render:
                    clear_output(wait=True)
                    self.env.render()
                
                self.prevActions = actions


    def generateActions(self, obs):
        ''' Generates a dictionary of actions. '''
        
        actions = {}

        for agent in self.env.agents:
            if agent.edge == None:
                targetNode = random.randint(0, len(self.env.pg.graph.nodes) - 1)
            else:
                targetNode = self.prevActions[agent]
            actions[agent] = targetNode
        
        return actions