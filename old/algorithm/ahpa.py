import networkx as nx
from IPython.display import clear_output

from .algorithm import BaseAlgorithm


class AHPA(BaseAlgorithm):
    ''' This algorithm implements AHPA for the patrolling problem. '''

    def save(self, *args, **kwargs):
        ''' Saves the model. '''
        pass


    def train(self, *args, seed=None, **kwargs):
        ''' Trains the model. '''
        pass


    def evaluate(self, render=False, render_terminal=False, max_cycles=None, max_episodes=1, seed=None):
        ''' Evaluates the model. '''
        
        if max_cycles != None:
            self.env.max_cycles = max_cycles

        for episode in range(max_episodes):
            obs, info = self.env.reset(seed=seed)

            # Assign AHPA info to each agent.
            for agent in self.env.agents:
                agent.ahpa = AhpaAgent(agent.id, self.env.pg, self.env.agentOrigins)

            if render:
                clear_output(wait=True)
                self.env.render()
            
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions = {agent: agent.ahpa.nodes[agent.ahpa.currentNodeIdx] for agent in self.env.agents}
                obs, rewards, terms, truncs, infos = self.env.step(actions)

                self.processObservation(obs)

                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]

                if render:
                    clear_output(wait=True)
                    self.env.render()

            if render_terminal:
                clear_output(wait=True)
                self.env.render()


    def processObservation(self, obs):
        ''' Returns the next node to visit. '''
        
        # Check whether any agents have reached their target node.
        for agent in self.env.agents:
            targetNode = agent.ahpa.nodes[agent.ahpa.currentNodeIdx]
            if agent.lastNode == targetNode and agent.edge == None:
                nextNode = agent.ahpa.getNextNode()


class AhpaAgent:
    def __init__(self, id, graph, agentOrigins):
        self.id = id
        self.graph = graph
        self.agentOrigins = agentOrigins

        # Set the allocation.
        self.voronoiOrigins = self.agentOrigins.copy()
        cell = self.getNodeAllocation(self.voronoiOrigins, self.agentOrigins)
        self.nodes = self.getNodeOrder(cell)
        self.currentNodeIdx = 1 if len(self.nodes) > 1 else 0
    

    def getNodeAllocation(self, origins, originalOrigins):
        ''' Returns the Voronoi partitions based on the origins provided. '''

        cells = nx.algorithms.voronoi.voronoi_cells(self.graph.graph, origins)
        return cells[originalOrigins[self.id]]


    def getNodeOrder(self, nodes):
        ''' Returns the visitation order for the provided nodes. '''

        if len(nodes) <= 1:
            return list(nodes)

        return nx.algorithms.approximation.traveling_salesman_problem(
            self.graph.graph,
            nodes=nodes,
            method=nx.algorithms.approximation.greedy_tsp
        )
    

    def getNextNode(self):
        ''' Returns the next node to visit. '''

        if self.currentNodeIdx >= len(self.nodes) - 1:
            self.currentNodeIdx = 0
        node = self.nodes[self.currentNodeIdx]
        if len(self.nodes) > 1:
            self.currentNodeIdx += 1
        return node