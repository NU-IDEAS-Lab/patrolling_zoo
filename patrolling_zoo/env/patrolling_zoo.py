from pettingzoo.utils.env import ParallelEnv
from patrolling_zoo.env.communication_model import CommunicationModel
from gymnasium import spaces
import random
import numpy as np
import math
from matplotlib import pyplot as plt
import networkx as nx
from copy import copy


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
     

class parallel_env(ParallelEnv):
    metadata = {
        "name": "patrolling_zoo_environment_v0",
    }

    def __init__(self, patrol_graph, num_agents,
                 comms_model = CommunicationModel(model = "bernoulli"),
                 require_explicit_visit = True,
                 speed = 1.0,
                 alpha = 10,
                 observation_radius = np.inf,
                 observe_method = "raw",
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
        self.observe_method = observe_method

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

        # Create the state space.
        state_space = {
            # The vertex state is composed of two parts.
            # The first part is the idleness time of each node.
            "vertex_state": spaces.Dict({
                v: spaces.Box(
                    low = 0.0,
                    high = np.inf,
                ) for v in range(self.pg.graph.number_of_nodes())
            }), # type: ignore
        }

        if self.observe_method != "ajg_new":
            # Add agent Euclidean position.
            state_space["agent_state"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([minPosX, minPosY], dtype=np.float32),
                    high = np.array([maxPosX, maxPosY], dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        if self.observe_method == "old":
            # The second part is the shortest path cost from every agent to every node.
            state_space["vertex_distances"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([0.0] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                    high = np.array([np.inf] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        elif self.observe_method == "ajg_new":
            state_space["vertex_distances"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([0.0] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                    high = np.array([np.inf] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        self.state_space = spaces.Dict(state_space)
        
        # Create the observation space.
        self.observation_spaces = spaces.Dict({agent: self.state_space for agent in self.possible_agents}) # type: ignore

        self.reset()


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
        
        return self.observe(self.possible_agents[0], radius=np.inf)

    def observe(self, agent, radius=None):
        ''' Returns the observation for the given agent.'''

        if radius == None:
            radius = agent.observationRadius

        agents = [a for a in self.agents if self._dist(a.position, agent.position) <= radius]
        vertices = [v for v in self.pg.graph.nodes if self._dist(self.pg.getNodePosition(v), agent.position) <= radius]

        if self.observe_method == "ranking":
            nodes_idless = {node : self.pg.getNodeIdlenessTime(node, self.step_count) for node in vertices}
            unique_sorted_idleness_times = sorted(list(set(nodes_idless.values())))
            obs = {
                "agent_state": {a: a.position for a in agents},
                "vertex_state": {v: unique_sorted_idleness_times.index(nodes_idless[v]) for v in vertices}
            }        
        elif self.observe_method == "normalization":
            nodes_idless = {node : self.pg.getNodeIdlenessTime(node, self.step_count) for node in vertices}
            min_ = min(nodes_idless.values())
            max_ = max(nodes_idless.values())
            obs = {
                "agent_state": {a: a.position for a in agents},
                "vertex_state": {v: (nodes_idless[v]-min_)/(max_ - min_) for v in vertices}
            }
        elif self.observe_method == "raw":
            nodes_idless = {node : self.pg.getNodeIdlenessTime(node, self.step_count) for node in vertices}
            obs = {
                "agent_state": {a: a.position for a in agents},
                "vertex_state": {v: nodes_idless[v] for v in vertices}
            }
        elif self.observe_method == "old":
            # Calculate the shortest path distances from each agent to each node.
            vertexDistances = {}
            for a in agents:
                vDists = np.zeros(self.pg.graph.number_of_nodes())
                for v in self.pg.graph.nodes:
                    path = self._getPathToNode(a, v)
                    vDists[v] = self._getAgentPathLength(a, path)
                vertexDistances[a] = vDists

            obs = {
                "agent_state": {a: a.position for a in agents},
                "vertex_state": {v: self.pg.getNodeIdlenessTime(v, self.step_count) for v in vertices},
                "vertex_distances": vertexDistances
            }
        elif self.observe_method == "ajg_new":
            # Calculate the shortest path distances from each agent to each node.
            vDists = np.zeros((len(agents), self.pg.graph.number_of_nodes()))
            for a in agents:
                for v in self.pg.graph.nodes:
                    path = self._getPathToNode(a, v)
                    vDists[a.id, v] = self._getAgentPathLength(a, path)
            
            # Normalize.
            if np.size(vDists) > 0:
                vDists = self._minMaxNormalize(vDists, minimum=0.0, maximum=self.pg.longestPathLength)

            # Convert to dictionary.
            vertexDistances = {}
            for a in agents:
                vertexDistances[a] = vDists[a.id]
            
            # Create numpy array of idleness times.
            idlenessTimes = np.zeros(self.pg.graph.number_of_nodes())
            for v in vertices:
                idlenessTimes[v] = self.pg.getNodeIdlenessTime(v, self.step_count)
            
            # Normalize.
            if np.size(idlenessTimes) > 0:
                if np.min(idlenessTimes) == np.max(idlenessTimes):
                    idlenessTimes = np.ones(self.pg.graph.number_of_nodes())
                else:
                    idlenessTimes = self._minMaxNormalize(idlenessTimes)

            obs = {
                "vertex_state": {v: idlenessTimes[v] for v in vertices},
                "vertex_distances": vertexDistances
            }
        
        else:
            raise ValueError(f"Invalid observation method {self.observe_method}")
        
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
                    
                    # Destination node is the action value.
                    dstNode = action

                    # Calculate the shortest path.
                    path = self._getPathToNode(agent, dstNode)
                    pathLen = self._getAgentPathLength(agent, path)
                    
                    # Provide reward for moving towards a node.
                    #idlenessDelta = self.pg.getNodeIdlenessTime(dstNode, self.step_count) - self.pg.getAverageIdlenessTime(self.step_count)
                    # idlenessDelta = self.pg.getNodeIdlenessTime(dstNode, self.step_count) / self.pg.getAverageIdlenessTime(self.step_count)
                    #if idlenessDelta >= 0:
                    #    r = self.alpha * idlenessDelta / (1.0 + np.log(1.0 + pathLen) / np.log(1000000))
                    #else:
                    #    r = self.alpha * idlenessDelta
                    #r = self.alpha * idlenessDelta
                    #reward_dict[agent] += r

                    # Take a step towards the next node.
                    stepSize = agent.speed
                    for nextNode in path:
                        reached, stepSize = self._moveTowardsNode(agent, nextNode, stepSize)

                        # The agent has reached the next node.
                        if reached:
                            if nextNode == dstNode or not self.requireExplicitVisit:
                                # The agent has reached its destination, visiting the node.
                                # The agent receives a reward for visiting the node.
                                r = self.onNodeVisit(nextNode, self.step_count)
                                #reward_dict[agent] += 100.0 * r
                
                        # The agent has exceeded its movement budget for this step.
                        if stepSize <= 0.0:
                            break
                else:
                    raise ValueError(f"Invalid action {action} for agent {agent.name}")

        # Assign the idleness penalty.
        # for agent in self.agents:
        #     # reward_dict[agent] -= np.log(self.pg.getStdDevIdlenessTime(self.step_count))
        #     reward_dict[agent] -= np.log(self.pg.getAverageIdlenessTime(self.step_count))
        #     #reward_dict[agent] -= np.log(self.pg.getWorstIdlenessTime(self.step_count))
        
        for agent in self.agents:
            reward_dict[agent] = self.step_count / (self.pg.getAverageIdlenessTime(self.step_count) + 1e-8)

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
            # Provide an end-of-episode reward.
            for agent in self.agents:
                reward_dict[agent] += 100.0 * self.max_cycles / (self.pg.getWorstIdlenessTime(self.step_count) + 1e-8)
                # reward_dict[agent] += 10000.0 / self.pg.getAverageIdlenessTime(self.step_count)
                # reward_dict[agent] /= self._minMaxNormalize(self.pg.getWorstIdlenessTime(self.step_count), minimum=0.0, maximum=self.max_cycles)
            
            truncated_dict = {a: True for a in self.agents}
            self.agents = []

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def onNodeVisit(self, node, timeStamp):
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''
        
        self.pg.setNodeVisitTime(node, timeStamp)
        return 0.0

        # avgIdleTime = self.pg.getAverageIdlenessTime(timeStamp)
        # self.pg.setNodeVisitTime(node, timeStamp)
        # deltaAvgIdleTime = avgIdleTime - self.pg.getAverageIdlenessTime(timeStamp)
        
        # return a reward which is proportional to the rank of the node, where the most idle node has the highest reward
        # return self.alpha * deltaAvgIdleTime

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
        
        # Set information about the node/edge which the agent is currently on.
        if reached:
            agent.lastNode = node
            agent.edge = None
        elif agent.lastNode != node:
            agent.edge = (agent.lastNode, node)

        return reached, max(stepSize - distCurrToNext, 0.0)


    def _dist(self, pos1, pos2):
        ''' Calculates the Euclidean distance between two points. '''

        return np.sqrt(np.power(pos1[0] - pos2[0], 2) + np.power(pos1[1] - pos2[1], 2))
    

    def _getPathToNode(self, agent, dstNode):
        ''' Determines the shortest path for the agent to reach the given node. '''

        path = []

        # The agent is on an edge, so determine which connected node results in shortest path.
        if agent.edge != None:
            path1 = nx.shortest_path(self.pg.graph, source=agent.edge[0], target=dstNode, weight='weight')
            pathLen1 = self._getAgentPathLength(agent, path1)
            path2 = nx.shortest_path(self.pg.graph, source=agent.edge[1], target=dstNode, weight='weight')
            pathLen2 = self._getAgentPathLength(agent, path2)
            path = path1
            if pathLen2 < pathLen1:
                path = path2
        
        # The agent is on a node. Simply calculate the shortest path.
        else:
            path = nx.shortest_path(self.pg.graph, source=agent.lastNode, target=dstNode, weight='weight')

            # Remove the first node from the path if the destination is different than the current node.
            if agent.lastNode != dstNode:
                path = path[1:]
        
        return path


    def _getAgentPathLength(self, agent, path):
        ''' Calculates the length of the given path for the given agent. '''

        pathLen = 0.0
        pathLen += self._dist(agent.position, self.pg.getNodePosition(path[0]))
        pathLen += nx.path_weight(self.pg.graph, path, weight='weight')

        return pathLen


    def _minMaxNormalize(self, x, eps=1e-8, a=0.0, b=1.0, maximum=None, minimum=None):
        ''' Normalizes numpy array x to be between a and b. '''

        if maximum is None:
            maximum = np.max(x)
        if minimum is None:
            minimum = np.min(x)
        return (x - minimum) / (maximum - minimum + eps)


    def observe_with_communication(self, agent):
        ''' Adds communicated states to the agent's observation. '''

        other_agents = [temp for temp in self.agents if temp != agent ]
        agent_observation = self.observe(agent)

        for a in other_agents:
            receive_obs = self.comms_model.canReceive(a, agent)

            if receive_obs:
                agent_observation["agent_state"][a] = a.position

        return agent_observation
