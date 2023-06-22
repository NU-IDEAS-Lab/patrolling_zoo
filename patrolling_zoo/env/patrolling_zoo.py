from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import random
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from copy import copy


class PatrolAgent():
    ''' This class stores all agent state. '''

    def __init__(self, id, position=(0.0, 0.0), speed=1.0, observationRadius=np.inf, startingNode=None):
        self.id = id
        self.name = f"agent_{id}"
        self.startingPosition = position
        self.startingSpeed = speed
        self.startingNode = startingNode
        self.observationRadius = observationRadius
        self.reset()
    
    
    def reset(self):
        self.position = self.startingPosition
        self.speed = self.startingSpeed
        self.edge = None
        self.lastNode = self.startingNode


class parallel_env(ParallelEnv):
    metadata = {
        "name": "patrolling_zoo_environment_v0",
    }

    def __init__(self, patrol_graph, num_agents,
                 require_explicit_visit = True,
                 observation_radius = np.inf,
                 max_steps: int = -1,
                 *args,
                 **kwargs):
        """
        Initialize the PatrolEnv object.

        Args:
            patrol_graph (PatrolGraph): The patrol graph representing the environment.
            num_agents (int): The number of agents in the environment.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

        self.pg = patrol_graph

        # Configuration.
        self.requireExplicitVisit = require_explicit_visit
        self.observationRadius = observation_radius
        self.maxSteps = max_steps

        # Create the agents with random starting positions.
        startingNodes = random.sample(self.pg.graph.nodes, num_agents)
        startingPositions = [self.pg.getNodePosition(i) for i in startingNodes]
        self.possible_agents = [
            PatrolAgent(i, startingPositions[i],
                        startingNode = startingNodes[i],
                        observationRadius = self.observationRadius
            ) for i in range(num_agents)
        ]

        # Create the action space.
        self.action_spaces = {agent: spaces.Discrete(len(self.pg.graph)) for agent in self.possible_agents}

        # Get graph bounds in Euclidean space.
        pos = nx.get_node_attributes(self.pg.graph, 'pos')
        minPosX = min(pos[p][0] for p in pos)
        maxPosX = max(pos[p][0] for p in pos)
        minPosY = min(pos[p][1] for p in pos)
        maxPosY = max(pos[p][1] for p in pos)

        # Create the observation space.
        self.observation_spaces = spaces.Dict({agent: spaces.Dict({

            # The agent state is the position of each agent.
            "agent_state": spaces.Dict({
                a: spaces.Box(
                    low = np.array([minPosX, minPosY]),
                    high = np.array([maxPosX, maxPosY]),
                ) for a in self.possible_agents
            }), # type: ignore

            # The vertex state is composed of two parts.
            # The first part is the idleness time of each node.
            "vertex_state": spaces.Dict({
                v: spaces.Box(
                    low = 0.0,
                    high = np.inf,
                ) for v in range(self.pg.graph.number_of_nodes())
            }), # type: ignore

            # The second part is the shortest path cost from every agent to every node.
            "vertex_distances": spaces.Dict({
                a: spaces.Box(
                    low = np.array([0.0] * self.pg.graph.number_of_nodes()),
                    high = np.array([np.inf] * self.pg.graph.number_of_nodes()),
                ) for a in self.possible_agents
            }) # type: ignore
        }) for agent in self.possible_agents}) # type: ignore

        self.reset()


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        # Reset the graph.
        self.pg.reset()

        # Reset the agents.
        self.agents = copy(self.possible_agents)
        for agent in self.possible_agents:
            agent.reset()
        
        # Reset other state.
        self.step_count = 0
        self.rewards = dict.fromkeys(self.agents, 0)
        self.dones = dict.fromkeys(self.agents, False)
        
        return {agent: self.observe(agent) for agent in self.agents}


    def render(self, figsize=(18, 12)):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).
                
            Returns:
                None
        '''
        fig, ax = plt.subplots(figsize=figsize)
        markers = ['p']
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']

        # Draw the graph.
        pos = nx.get_node_attributes(self.pg.graph, 'pos')
        idleness = [self.pg.getNodeIdlenessTime(i, self.step_count) for i in self.pg.graph.nodes]
        nx.draw_networkx(self.pg.graph,
                         pos,
                         with_labels=True,
                         node_color=idleness,
                         edgecolors='black',
                         vmin=0,
                         vmax=50,
                         cmap='Purples',
                         node_size=600,
                         font_size=10,
                         font_color='black'
        )
        nx.draw_networkx_edge_labels(self.pg.graph, pos, edge_labels=nx.get_edge_attributes(self.pg.graph, 'weight'), font_size=7)
        
        # Draw the agents.
        for i, agent in enumerate(self.agents):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            plt.scatter(*agent.position, color=colors[i % len(colors)], marker=markers[i % len(markers)], zorder=10, alpha=0.3, s=300)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=agent.name, alpha=0.5)

        plt.legend()
        plt.text(0,0,f'Current step: {self.step_count}, Average idleness time: {self.pg.getAverageIdlenessTime(self.step_count)}')
        plt.show()


    def observation_space(self, agent):
        ''' Returns the observation space for the given agent. '''
        return self.observation_spaces[agent]


    def action_space(self, agent):
        ''' Returns the action space for the given agent. '''
        return self.action_spaces[agent]

    def state(self):
        ''' Returns the global state of the environment.
            This is useful for centralized training, decentralized execution. '''
        
        raise NotImplementedError()

    def observe(self, agent):
        ''' Returns the observation for the given agent.'''

        agents = [a for a in self.agents if self._dist(a.position, agent.position) <= agent.observationRadius]
        vertices = [v for v in self.pg.graph.nodes if self._dist(self.pg.getNodePosition(v), agent.position) <= agent.observationRadius]

        # Calculate the shortest path distances from each agent to each node.
        vertexDistances = {}
        for a in agents:
            vDists = nx.shortest_path_length(self.pg.graph,
                                  source=self.pg.getNearestNode(a.position),
                                  weight='weight'
            )
            vertexDistances[a] = np.array([vDists[v] for v in range(self.pg.graph.number_of_nodes())])

        return {
            "agent_state": {a: a.position for a in agents},
            "vertex_state": {v: self.pg.getNodeIdlenessTime(v, self.step_count) for v in vertices},
            "vertex_distances": vertexDistances
        }


    def step(self, action_dict={}):
        ''''
        Perform a step in the environment based on the given action dictionary.

        Args:
            action_dict (dict): A dictionary containing actions for each agent.

        Returns:
            obs_dict (dict): A dictionary containing the observations for each agent.
            reward_dict (dict): A dictionary containing the rewards for each agent.
            done_dict (dict): A dictionary indicating whether each agent is done.
            info_dict (dict): A dictionary containing additional information for each agent.
        '''
        self.step_count += 1
        obs_dict = {}
        reward_dict = {agent: 0.0 for agent in self.agents}
        done_dict = {}
        truncated_dict = {agent: False for agent in self.agents}
        info_dict = {}

        # Perform actions.
        for agent in self.agents:
            # If the agent is at a node, not transitioning
            if agent in action_dict:
                action = action_dict[agent]

                # Update the agent's position.
                if action in self.pg.graph.nodes:

                    # Determine the node to use as source node for shortest path calculation.
                    startIdx = 1
                    srcNode = agent.lastNode
                    dstNode = action
                    # The agent is on an edge, so determine which connected node results in shortest path.
                    if agent.edge != None:
                        pathLen1 = nx.shortest_path_length(self.pg.graph, source=agent.edge[0], target=dstNode, weight='weight')
                        pathLen2 = nx.shortest_path_length(self.pg.graph, source=agent.edge[1], target=dstNode, weight='weight')
                        srcNode = agent.edge[0]
                        if pathLen2 < pathLen1:
                            srcNode = agent.edge[1]
                        startIdx = 0
                    
                    # Calculate the shortest path.
                    path = nx.shortest_path(self.pg.graph, source=srcNode, target=dstNode, weight='weight')
                    
                    # Handle special case where the agent is already at the destination node.
                    if srcNode == dstNode:
                        startIdx = 0

                    # Take a step towards the next node.
                    stepSize = agent.speed
                    for nextNode in path[startIdx:]:
                        # print(f"Moving towards next node {nextNode} with step size {stepSize}")
                        reached, stepSize = self._moveTowardsNode(agent, nextNode, stepSize)

                        # The agent has reached the next node.
                        if reached:
                            agent.lastNode = nextNode
                            if agent.lastNode == dstNode or not self.requireExplicitVisit:
                                # The agent has reached its destination, visiting the node.
                                # The agent receives a reward for visiting the node.
                                reward_dict[agent] += self.onNodeVisit(agent, agent.lastNode, self.step_count)

                        # The agent has exceeded its movement budget for this step.
                        if stepSize <= 0.0:
                            break
                else:
                    reward_dict[agent] = 0  # the action was invalid

        # Perform observations.
        for agent in self.agents:
            # Check if the agent is done
            done_dict[agent] = self.dones[agent]

            # Check for truncation
            if not self.dones[agent]:
                truncated_dict[agent] = self.step_count >= self.maxSteps if self.maxSteps is not None else False

            # Add any additional information for the agent
            info_dict[agent] = {}

            # Update the observation for the agent
            obs_dict[agent] = self.observe(agent)

        # Check truncation conditions.
        if self.maxSteps >= 0 and self.step_count >= self.maxSteps:
            truncated_dict = {a: True for a in self.agents}
            self.agents = []

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def onNodeVisit(self, agent, node, timeStamp):
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''

        idleTime = self.pg.getNodeIdlenessTime(node, timeStamp)
        self.pg.setNodeVisitTime(node, timeStamp)
        return idleTime


    def _moveTowardsNode(self, agent, node, stepSize):
        ''' Takes a single step towards the next node.
            Returns a tuple containing whether the agent has reached the node
            and the remaining step size. '''

        # Take a step towards the next node.
        posNextNode = self.pg.getNodePosition(node)
        distCurrToNext = self._dist(agent.position, posNextNode)
        reached = distCurrToNext <= stepSize
        step = distCurrToNext if reached else stepSize
        if distCurrToNext > 0.0:
            agent.position = (agent.position[0] + (posNextNode[0] - agent.position[0]) * step / distCurrToNext,
                                agent.position[1] + (posNextNode[1] - agent.position[1]) * step / distCurrToNext)
        
        # Set information about the edge which the agent is currently on.
        if reached:
            agent.edge = None
        else:
            agent.edge = (agent.lastNode, node)

        return reached, stepSize - distCurrToNext


    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.sqrt(np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2))