from pettingzoo.utils.env import ParallelEnv
from patrolling_zoo.env.communication_model import CommunicationModel
from gymnasium import spaces
import random
import numpy as np
import math
from matplotlib import pyplot as plt
import networkx as nx
from copy import copy
from enum import IntEnum


class PatrolAgent():
    ''' This class stores all agent state. '''

    def __init__(self, id, position=(0.0, 0.0), speed = 1.0, observationRadius=np.inf, startingNode=None, currentState = 1):
        self.id = id
        self.name = f"agent_{id}"
        self.startingPosition = position
        self.startingSpeed = speed
        self.startingNode = startingNode
        self.observationRadius = observationRadius
        self.currentState = currentState
        self.reset()
    
    
    def reset(self):
        self.position = self.startingPosition
        self.speed = self.startingSpeed
        self.edge = None
        self.lastNode = self.startingNode
        self.lastAction = None
        self.stepsTravelled = 0
     

class parallel_env(ParallelEnv):
    metadata = {
        "name": "patrolling_zoo_environment_v0",
    }

    class OBSERVATION_CHANNELS(IntEnum):
        AGENT_ID = 0
        IDLENESS = 1
        OBSTACLE = 2

    def __init__(self, patrol_graph, num_agents,
                 comms_model = CommunicationModel(model = "bernoulli"),
                 require_explicit_visit = True,
                 speed = 1.0,
                 alpha = 10,
                 observation_radius = np.inf,
                 observation_mode = "vector",
                 max_cycles: int = -1,
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
        self.max_cycles = max_cycles
        self.comms_model = comms_model
        self.observationMode = observation_mode

        self.alpha = alpha

        # Create the agents with random starting positions.
        startingNodes = random.sample(list(self.pg.graph.nodes), num_agents)
        startingPositions = [self.pg.getNodePosition(i) for i in startingNodes]
        self.possible_agents = [
            PatrolAgent(i, startingPositions[i],
                        speed = speed,
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
        if self.observationMode == "bitmap":
            observation_space = spaces.Box(
                low=-1.0,
                high=np.inf,
                shape=(self.pg.widthPixels, self.pg.heightPixels, len(self.OBSERVATION_CHANNELS)),# + self.pg.graph.number_of_nodes()),
                dtype=np.float32,
            )
        elif self.observationMode == "vector":
            observation_space = spaces.Dict({
                # The agent state is the position of each agent.
                "agent_state": spaces.Dict({
                    a: spaces.Box(
                        low = np.array([minPosX, minPosY], dtype=np.float32),
                        high = np.array([maxPosX, maxPosY], dtype=np.float32),
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
                        low = np.array([0.0] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                        high = np.array([np.inf] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                    ) for a in self.possible_agents
                }) # type: ignore
            })
        else:
            raise ValueError(f"Invalid observation mode {self.observationMode}")
        
        self.observation_spaces = spaces.Dict({
            agent: observation_space for agent in self.possible_agents
        })


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        if seed != None:
            random.seed(seed)

        # Reset the graph.
        self.pg.reset()

        # Reset the agents.
        startingNodes = random.sample(list(self.pg.graph.nodes), len(self.possible_agents))
        startingPositions = [self.pg.getNodePosition(i) for i in startingNodes]
        self.agents = copy(self.possible_agents)
        for agent in self.possible_agents:
            agent.startingPosition = startingPositions[agent.id]
            agent.startingNode = startingNodes[agent.id]
            agent.reset()
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)
        
        observation = {agent: self.observe(agent) for agent in self.agents}
        info = {}
        return observation, info


    def render(self, figsize=(12, 9)):
        ''' Renders the environment.
            
            Args:
                figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).
                
            Returns:
                None
        '''
        fig, ax = plt.subplots(figsize=figsize)
        markers = ['p']
        markers_done = ['X']
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
        weights = {key: np.round(value, 1) for key, value in nx.get_edge_attributes(self.pg.graph, 'weight').items()}
        nx.draw_networkx_edge_labels(self.pg.graph, pos, edge_labels=weights, font_size=7)
        
        # Draw the agents.
        for i, agent in enumerate(self.possible_agents):
            marker = markers[i % len(markers)] if agent in self.agents else markers_done[i % len(markers_done)]
            color = colors[i % len(colors)]
            plt.scatter(*agent.position, color=color, marker=marker, zorder=10, alpha=0.3, s=300)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=agent.name, alpha=0.5)

        plt.legend()
        plt.gcf().text(0,0,f'Current step: {self.step_count}, Average idleness time: {self.pg.getAverageIdlenessTime(self.step_count)}')
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
        
        return self.observe(self.agents[0])


    def observe(self, agent):
        ''' Returns the observation for the given agent.'''

        agents = [a for a in self.agents if self._dist(a.position, agent.position) <= agent.observationRadius]
        vertices = [v for v in self.pg.graph.nodes if self._dist(self.pg.getNodePosition(v), agent.position) <= agent.observationRadius]

        if self.observationMode == "bitmap":
            # Calculate the shortest path distances from each agent to each node.
            obs = -1.0 * np.ones(self.observation_space(agent).shape, dtype=np.float32)

            # Add agents to the observation.
            for a in agents:
                obs[int(a.position[0]), int(a.position[1]), self.OBSERVATION_CHANNELS.AGENT_ID] = a.id
            
            # Add vertices to the observation.
            for v in vertices:
                pos = self.pg.getNodePosition(v)
                obs[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.IDLENESS] = self.pg.getNodeIdlenessTime(v, self.step_count)
            
            # Create a connectivity matrix for each vertex and then add it to the remaining channels in the observation.
            # We assume that the agent always knows the graph in advance.
            # for v in self.pg.graph.nodes:
            #     pos = self.pg.getNodePosition(v)
            #     edges = self.pg.graph.edges(v)
            #     for _, neighbor in edges:
            #         obs[int(pos[0]), int(pos[1]), len(self.OBSERVATION_CHANNELS) + neighbor] = 1.0

            # Add a connectivity channel which "draws" lines for each edge.
            for edge in self.pg.graph.edges:
                pos1 = self.pg.getNodePosition(edge[0])
                pos2 = self.pg.getNodePosition(edge[1])
                dist = self._dist(pos1, pos2)
                if dist > 0.0:
                    for i in range(int(dist)):
                        obs[int(pos1[0] + (pos2[0] - pos1[0]) * i / dist), int(pos1[1] + (pos2[1] - pos1[1]) * i / dist), self.OBSERVATION_CHANNELS.OBSTACLE] = 1.0

            # Fancier edge drawing.
            # for edge in self.pg.graph.edges:
            #     pos1 = np.array(self.pg.getNodePosition(edge[0]))
            #     pos2 = np.array(self.pg.getNodePosition(edge[1]))
            #     for i in range(obs.shape[0]):
            #         for j in range(obs.shape[1]):
            #             pos3 = np.array((i, j))
            #             d = np.cross(pos2 - pos1, pos3 - pos1) / np.linalg.norm(pos2 - pos1)
            #             if d < 5.0:
            #                 obs[i, j, self.OBSERVATION_CHANNELS.OBSTACLE] = 0.0

        elif self.observationMode == "vector":
            # Calculate the shortest path distances from each agent to each node.
            vertexDistances = {}
            for a in agents:
                vDists = nx.shortest_path_length(self.pg.graph,
                                    source=self.pg.getNearestNode(a.position),
                                    weight='weight'
                )
                vertexDistances[a] = np.array([vDists[v] for v in self.pg.graph.nodes])

            # Create the observation.
            obs = {
                "agent_state": {a: a.position for a in agents},
                "vertex_state": {v: self.pg.getNodeIdlenessTime(v, self.step_count) for v in vertices},
                "vertex_distances": vertexDistances
            }
        
        else:
            raise ValueError(f"Invalid observation mode {self.observationMode}")

        return obs


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
        reward_dict = {agent: 0 for agent in self.agents}
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

                    # if action == self.pg.graph.number_of_nodes() - 2:
                    #     reward_dict[agent] += 1000

                    if agent.lastAction != action:
                        agent.lastAction = action
                        agent.stepsTravelled = 0

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
                            same = agent.lastNode == nextNode
                            agent.lastNode = nextNode
                            if agent.lastNode == dstNode or not self.requireExplicitVisit:
                                # The agent has reached its destination, visiting the node.
                                # The agent receives a reward for visiting the node.
                                r = self.onNodeVisit(agent.lastNode, self.step_count)
                                if not same:
                                    # print(f"REACHED DESTINATION NODE {dstNode} WITH REWARD {r}")
                                    reward_dict[agent] += 100.0 * r
                
                        # The agent has exceeded its movement budget for this step.
                        if stepSize <= 0.0:
                            break
                    
                    if srcNode != dstNode:
                        if agent.stepsTravelled > 1:
                            reward_dict[agent] += np.log(agent.stepsTravelled)
                        agent.stepsTravelled += 1
                else:
                    raise ValueError(f"Invalid action {action} for agent {agent.name}")

        # Assign the idleness penalty.
        # for agent in self.agents:
        #     # reward_dict[agent] -= np.log(self.pg.getStdDevIdlenessTime(self.step_count))
        #     reward_dict[agent] -= np.log(self.pg.getAverageIdlenessTime(self.step_count))
        #     #reward_dict[agent] -= np.log(self.pg.getWorstIdlenessTime(self.step_count))
        
        # for agent in self.agents:
        #     reward_dict[agent] = (self.pg.graph.number_of_nodes() * self.step_count) / (self.pg.getAverageIdlenessTime(self.step_count) + 0.0001)

        # Perform observations.
        for agent in self.agents:

            # 3 communicaiton models here
            # agent_observation = self.observe_with_communication(agent)
            agent_observation = self.observe(agent)
            
            # Check if the agent is done
            done_dict[agent] = self.dones[agent]

            # Add any additional information for the agent
            info_dict[agent] = {}

            # Update the observation for the agent
            obs_dict[agent] = agent_observation

        # Check truncation conditions.
        if self.max_cycles >= 0 and self.step_count >= self.max_cycles:
            truncated_dict = {a: True for a in self.agents}
            self.agents = []

            reward_dict[agent] += 10000.0 / self.pg.getWorstIdlenessTime(self.step_count)
            # reward_dict[agent] += 10000.0 / self.pg.getAverageIdlenessTime(self.step_count)

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def onNodeVisit(self, node, timeStamp):
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''

        # idleTime = self.pg.getNodeIdlenessTime(node, timeStamp) 
        # self.pg.setNodeVisitTime(node, timeStamp)
        # return idleTime

        # Here we rank the nodes in term of idleness and give a reward based on the rank.
        # So the agent will be encouraged to visit the most idle node.
        # nodes_idless = {node : self.pg.getNodeIdlenessTime(node, self.step_count) for node in self.pg.graph.nodes}
        # indices = sorted(nodes_idless, key=nodes_idless.get)

        # nodesSorted = sorted(self.pg.graph.nodes(), key=lambda n: self.pg.graph.nodes[n]['visitTime'])
        # index = nodesSorted.index(node)

        # print(f"IDLENESS TIME ORDERED: {nodesSorted}, {[self.pg.getNodeIdlenessTime(n, self.step_count) for n in nodesSorted]}")

        avgIdleTime = self.pg.getAverageIdlenessTime(timeStamp)
        self.pg.setNodeVisitTime(node, timeStamp)
        deltaAvgIdleTime = avgIdleTime - self.pg.getAverageIdlenessTime(timeStamp)
        
        # return a reward which is proportional to the rank of the node, where the most idle node has the highest reward
        return self.alpha * deltaAvgIdleTime

        # return self.alpha ** max((index - self.reward_shift * len(indices))/self.pg.graph.number_of_nodes(), 0)


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
            agent.position = (
                agent.position[0] + (posNextNode[0] - agent.position[0]) * step / distCurrToNext,
                agent.position[1] + (posNextNode[1] - agent.position[1]) * step / distCurrToNext
            )
        
        # Set information about the edge which the agent is currently on.
        if reached:
            agent.edge = None
        else:
            agent.edge = (agent.lastNode, node)

        return reached, stepSize - distCurrToNext


    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.sqrt(np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2))
    

    def observe_with_communication(self, agent):
        ''' Adds communicated states to the agent's observation. '''

        other_agents = [temp for temp in self.agents if temp != agent ]
        agent_observation = self.observe(agent)

        for a in other_agents:
            receive_obs = self.comms_model.canReceive(a, agent)

            if receive_obs:
                agent_observation["agent_state"][a] = a.position

        return agent_observation
