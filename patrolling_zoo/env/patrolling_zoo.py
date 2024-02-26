from pettingzoo.utils.env import ParallelEnv
from patrolling_zoo.env.communication_model import CommunicationModel
from gymnasium import spaces
import random
import numpy as np
import math
from copy import deepcopy
from matplotlib import pyplot as plt
import networkx as nx
from copy import copy
from enum import IntEnum
from torch_geometric.utils.convert import from_networkx
from torch_geometric.data import Data


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
        self.currentAction = -1.0
        self.lastNode = self.startingNode
        self.lastNodeVisited = None
     

class parallel_env(ParallelEnv):
    metadata = {
        "name": "patrolling_zoo_environment_v0",
    }

    class OBSERVATION_CHANNELS(IntEnum):
        AGENT_ID = 0
        IDLENESS = 1
        GRAPH = 2

    def __init__(self, patrol_graph, num_agents,
                 comms_model = CommunicationModel(model = "none"),
                 require_explicit_visit = True,
                 speed = 1.0,
                 alpha = 10.0,
                 beta = 100.0,
                 action_method = "full",
                 action_full_max_nodes = 40,
                 action_neighbors_max_degree = 15,
                 reward_method_terminal = "average",
                 observation_radius = np.inf,
                 observe_method = "ajg_new",
                 observe_method_global = None,
                 observe_bitmap_dims = (50, 50),
                 attrition_method = "none",
                 attrition_random_probability = 0.0,
                 attrition_min_agents = 2,
                 attrition_times = [],
                 max_cycles: int = -1,
                 reward_interval: int = -1,
                 regenerate_graph_on_reset: bool = False,
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
        self.action_method = action_method
        self.action_full_max_nodes = action_full_max_nodes
        self.action_neighbors_max_degree = action_neighbors_max_degree
        self.reward_method_terminal = reward_method_terminal
        self.observe_method = observe_method
        self.observe_method_global = observe_method_global if observe_method_global != None else observe_method
        self.observe_bitmap_dims = observe_bitmap_dims
        self.attrition_method = attrition_method
        self.attrition_random_probability = attrition_random_probability
        self.attrition_times = attrition_times
        self.attrition_min_agents = attrition_min_agents
        self.regenerate_graph_on_reset = regenerate_graph_on_reset

        self.reward_interval = reward_interval

        self.alpha = alpha
        self.beta = beta

        # Create the agents with random starting positions.
        self.agentOrigins = random.sample(list(self.pg.graph.nodes), num_agents)
        startingPositions = [self.pg.getNodePosition(i) for i in self.agentOrigins]
        self.possible_agents = [
            PatrolAgent(i, startingPositions[i],
                        speed = speed,
                        startingNode = self.agentOrigins[i],
                        observationRadius = self.observationRadius
            ) for i in range(num_agents)
        ]

        # Create the action space.
        action_space = self._buildActionSpace(self.action_method)
        self.action_spaces = spaces.Dict({agent: action_space for agent in self.possible_agents}) # type: ignore

        # The state space is a complete observation of the environment.
        # This is not part of the standard PettingZoo API, but is useful for centralized training.
        self.state_space = self._buildStateSpace(self.observe_method_global)
        
        # Create the observation space.
        obs_space = self._buildStateSpace(self.observe_method)
        self.observation_spaces = spaces.Dict({agent: obs_space for agent in self.possible_agents}) # type: ignore

        self.reset_count = 0
        self.reset()


    def _buildActionSpace(self, action_method):
        ''' Creates a gym.spaces.* object representing the action space. '''

        if action_method == "full":
            if self.action_full_max_nodes < len(self.pg.graph):
                raise ValueError("The action space is smaller than the graph size.")
            maxNodes = self.action_full_max_nodes if self.action_full_max_nodes > 0 else len(self.pg.graph)
            return spaces.Discrete(maxNodes)
        
        elif action_method == "neighbors":
            # we subtract 1 since for some reason the self-loop increases degree by 2
            # maxDegree = max([self.pg.graph.degree(node) - 1 for node in self.pg.graph.nodes])
            # maxDegree = max([self.pg.graph.degree(node) for node in self.pg.graph.nodes])
            maxDegree = self.action_neighbors_max_degree # just use a fixed size and mask it
            return spaces.Discrete(maxDegree)


    def _buildStateSpace(self, observe_method):
        ''' Creates a state space given the observation method.
            Returns a gym.spaces.* object. '''
        
        # Create the state space dictionary.
        state_space = {}

        # Add to the dictionary depending on the observation method.

        # Add agent id.
        if observe_method in ["ajg_new", "ajg_newer", "adjacency", "pyg"]:
            state_space["agent_id"] = spaces.Box(
                low = -1,
                high = len(self.possible_agents),
                dtype=np.int32
            )

        # Add vertex idleness time.
        if observe_method in ["ranking", "raw", "old", "ajg_new", "ajg_newer", "adjacency", "idlenessOnly"]:
            state_space["vertex_state"] = spaces.Dict({
                v: spaces.Box(
                    low = -1.0,
                    high = np.inf,
                ) for v in range(self.pg.graph.number_of_nodes())
            }) # type: ignore

        # Add agent Euclidean position.
        if observe_method in ["ranking", "raw", "old"]:
            # Get graph bounds in Euclidean space.
            pos = nx.get_node_attributes(self.pg.graph, 'pos')
            minPosX = min(pos[p][0] for p in pos)
            maxPosX = max(pos[p][0] for p in pos)
            minPosY = min(pos[p][1] for p in pos)
            maxPosY = max(pos[p][1] for p in pos)

            state_space["agent_state"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([minPosX, minPosY], dtype=np.float32),
                    high = np.array([maxPosX, maxPosY], dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        # Add vertex distances from each agent.
        if observe_method in ["old", "ajg_new", "ajg_newer"]:
            state_space["vertex_distances"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([0.0] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                    high = np.array([np.inf] * self.pg.graph.number_of_nodes(), dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        # Add bitmap observation.
        if observe_method in ["bitmap", "bitmap2"]:
            state_space = spaces.Box(
                low=-2.0,
                high=np.inf,
                shape=(self.observe_bitmap_dims[0], self.observe_bitmap_dims[1], len(self.OBSERVATION_CHANNELS)),
                dtype=np.float32,
            )
        
        # Add adjacency matrix.
        if observe_method in ["adjacency", "ajg_newer"]:
            state_space["adjacency"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes()),
                dtype=np.float32,
            )
        
        # Add agent graph position vector.
        if observe_method in ["adjacency", "ajg_newer"]:
            state_space["agent_graph_position"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                    high = np.array([self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes(), 1.0], dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        if observe_method in ["pyg"]:
            if self.action_method == "neighbors":
                edge_space = spaces.Box(
                    # weight, neighborID
                    low = np.array([0.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf], dtype=np.float32),
                )
                # node_space = spaces.Box(
                #     # nodeType,visitTime
                #     low = np.array([-np.inf, 0.0], dtype=np.float32),
                #     high = np.array([np.inf, np.inf], dtype=np.float32),
                # )
                # node_type_idx = 0
                node_space = spaces.Box(
                    # ID, nodeType,visitTime, lastNode
                    low = np.array([0.0, -np.inf, 0.0, -1.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                )
                node_type_idx = 1
            else:
                edge_space = spaces.Box(
                    # weight
                    low = np.array([0.0], dtype=np.float32),
                    high = np.array([np.inf], dtype=np.float32),
                )
                node_space = spaces.Box(
                    # ID, nodeType,visitTime, lastNode, currentAction
                    low = np.array([0.0, -np.inf, 0.0, -1.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                )
                node_type_idx = 1

            state_space["graph"] = spaces.Graph(
                node_space = node_space,
                edge_space = edge_space
            )
            state_space["graph"].node_type_idx = node_type_idx

            # state_space["lastNode"] = spaces.Box(
            #     low = 0,
            #     high = self.pg.graph.number_of_nodes(),
            #     dtype=np.int32
            # )
            # state_space["currentAction"] = spaces.Box(
            #     low = -1.0,
            #     high = np.inf,
            #     dtype=np.float32
            # )
        
        if type(state_space) == dict:
            state_space = spaces.Dict(state_space)
        
        return state_space


    def reset(self, seed=None, options=None):
        ''' Sets the environment to its initial state. '''

        self.reset_count += 1

        if seed != None:
            random.seed(seed)

        # Reset the graph.
        regenerateGraph = self.regenerate_graph_on_reset and self.reset_count % 20 == 0
        randomizeIds = regenerateGraph
        self.pg.reset(seed, randomizeIds=randomizeIds, regenerateGraph=regenerateGraph)

        # Reset the information about idleness over time.
        self.avgIdlenessTimes = []

        # Reset the node visit counts.
        self.nodeVisits = np.zeros(self.pg.graph.number_of_nodes())

        # Reset the agents.
        self.agentOrigins = random.sample(list(self.pg.graph.nodes), len(self.possible_agents))
        startingPositions = [self.pg.getNodePosition(i) for i in self.agentOrigins]
        self.agents = copy(self.possible_agents)
        for agent in self.possible_agents:
            agent.startingPosition = startingPositions[agent.id]
            agent.startingNode = self.agentOrigins[agent.id]
            agent.reset()
        
        # Reset other state.
        self.step_count = 0
        self.dones = dict.fromkeys(self.agents, False)

        # Set available actions.
        self.available_actions = {agent: self._getAvailableActions(agent) for agent in self.agents}

        # Return the initial observation.
        observation = {agent: self.observe(agent) for agent in self.agents}
        info = {
            agent: {
                "ready": True
            } for agent in self.agents
        }

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
        nodeColors = [self._minMaxNormalize(idleness[i], a=0.0, b=100, minimum=0.0, maximum=self.step_count) for i in self.pg.graph.nodes]
        nx.draw_networkx(self.pg.graph,
                         pos,
                         with_labels=True,
                         node_color=idleness,
                         edgecolors='black',
                         vmin=0,
                         vmax=100,
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

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.gcf().text(0,0,f'Current step: {self.step_count}, Average idleness time: {self.pg.getAverageIdlenessTime(self.step_count):.2f}')
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
        
        return self._populateStateSpace(self.observe_method_global, self.possible_agents[0], radius=np.inf, allow_done_agents=True)

    def state_all(self):
        ''' Similar to the state() method, but this returns a customized copy of the state space for each agent.
            This is useful for centralized training, decentralized execution. '''
        
        state = {}
        for agent in self.possible_agents:
            state[agent] = self._populateStateSpace(self.observe_method_global, agent, radius=np.inf, allow_done_agents=True)
        return state


    def observe(self, agent, radius=None, allow_done_agents=False):
        ''' Returns the observation for the given agent.'''

        return self._populateStateSpace(self.observe_method, agent, radius, allow_done_agents)


    def _populateStateSpace(self, observe_method, agent, radius, allow_done_agents):
        ''' Returns a populated state/observation space.'''

        if radius == None:
            radius = agent.observationRadius

        if allow_done_agents:
            agentList = self.possible_agents
        else:
            agentList = self.agents

        # Calculate the list of visible agents and vertices.
        vertices = [v for v in self.pg.graph.nodes if self._dist(self.pg.getNodePosition(v), agent.position) <= radius]
        agents = [a for a in agentList if self._dist(a.position, agent.position) <= radius]
        for a in agentList:
            if a != agent and a not in agents and self.comms_model.canReceive(a, agent):
                agents.append(a)
                for v in self.pg.graph.nodes:
                    if v not in vertices and self._dist(self.pg.getNodePosition(v), a.position) <= radius:
                        vertices.append(v)
        agents = sorted(agents, key=lambda a: a.id)
        vertices = sorted(vertices)
        
        obs = {}

        # Add agent ID.
        if observe_method in ["ajg_new", "ajg_newer", "adjacency", "pyg"]:
            obs["agent_id"] = agent.id

        # Add agent position.
        if observe_method in ["ranking", "raw", "old"]:
            obs["agent_state"] = {a: a.position for a in agents}

        # Add vertex idleness time (ranked).
        if observe_method in ["ranking"]:
            nodes_idless = {node : self.pg.getNodeIdlenessTime(node, self.step_count) for node in vertices}
            unique_sorted_idleness_times = sorted(list(set(nodes_idless.values())))
            obs["vertex_state"] = {v: unique_sorted_idleness_times.index(nodes_idless[v]) for v in vertices}
        
        # Add vertex idleness time (minMax normalized).
        if observe_method in ["ajg_new", "ajg_newer"]:
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

            # Create dictionary with default value of -1.0.
            obs["vertex_state"] = {v: -1.0 for v in range(self.pg.graph.number_of_nodes())}

            # Fill actual values for nodes we can see.
            for v in vertices:
                obs["vertex_state"][v] = idlenessTimes[v]

        # Add vertex idleness time (raw).
        if observe_method in ["raw", "old", "idlenessOnly", "adjacency"]:
            # Create dictionary with default value of -1.0.
            obs["vertex_state"] = {v: -1.0 for v in range(self.pg.graph.number_of_nodes())}

            for node in vertices:
                obs["vertex_state"][node] = self.pg.getNodeIdlenessTime(node, self.step_count)

        # Add vertex distances from each agent (raw).
        if observe_method in ["old"]:
            vertexDistances = {}
            for a in agents:
                vDists = np.zeros(self.pg.graph.number_of_nodes())
                for v in self.pg.graph.nodes:
                    path = self._getPathToNode(a, v)
                    vDists[v] = self._getAgentPathLength(a, path)
                vertexDistances[a] = vDists
            obs["vertex_distances"] = vertexDistances

        # Add vertex distances from each agent (normalized).
        if observe_method in ["ajg_new", "ajg_newer"]:
            # Calculate the shortest path distances from each agent to each node.
            vDists = np.ones((len(self.possible_agents), self.pg.graph.number_of_nodes()))
            for a in agents:
                for v in self.pg.graph.nodes:
                    path = self._getPathToNode(a, v)
                    dist = self._getAgentPathLength(a, path)
                    dist = self._minMaxNormalize(dist, minimum=0.0, maximum=self.pg.longestPathLength)
                    vDists[a.id, v] = dist
            
            # Convert to dictionary.
            vertexDistances = {}
            for a in self.possible_agents:
                vertexDistances[a] = vDists[a.id]
            
            obs["vertex_distances"] = vertexDistances

        # Add bitmap observation.
        if observe_method in ["bitmap"]:
            # Create an image which defaults to -1.
            bitmap = -1.0 * np.ones(self.observation_space(agent).shape, dtype=np.float32)

            # Set the observing agent's ID in the (0, 0) position. This is a bit hacky.
            bitmap[0, 0, self.OBSERVATION_CHANNELS.AGENT_ID] = agent.id

            def _normPosition(pos):
                if radius == np.inf:
                    x = self._minMaxNormalize(pos[0], a=0.0, b=self.observe_bitmap_dims[0], minimum=0.0, maximum=self.pg.widthPixels, eps=0.01)
                    y = self._minMaxNormalize(pos[1], a=0.0, b=self.observe_bitmap_dims[1], minimum=0.0, maximum=self.pg.heightPixels, eps=0.01)
                else:
                    x = self._minMaxNormalize(pos[0], a=0.0, b=self.observe_bitmap_dims[0], minimum=agent.position[0] - radius, maximum=agent.position[0] + radius, eps=0.01)
                    y = self._minMaxNormalize(pos[1], a=0.0, b=self.observe_bitmap_dims[1], minimum=agent.position[1] - radius, maximum=agent.position[1] + radius, eps=0.01)
                return x, y

            # Add agents to the observation.
            for a in agents:
                pos = _normPosition(a.position)
                if pos[0] < 0 or pos[0] >= self.observe_bitmap_dims[0] or pos[1] < 0 or pos[1] >= self.observe_bitmap_dims[1]:
                    continue
                bitmap[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.AGENT_ID] = a.id
            
            # Add vertex idleness times to the observation.
            for v in vertices:
                pos = _normPosition(self.pg.getNodePosition(v))
                if pos[0] < 0 or pos[0] >= self.observe_bitmap_dims[0] or pos[1] < 0 or pos[1] >= self.observe_bitmap_dims[1]:
                    continue
                bitmap[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.IDLENESS] = self.pg.getNodeIdlenessTime(v, self.step_count)
            
            # Add edges to the graph channel.
            for edge in self.pg.graph.edges:
                pos1 = _normPosition(self.pg.getNodePosition(edge[0]))
                pos2 = _normPosition(self.pg.getNodePosition(edge[1]))
                dist = self._dist(pos1, pos2)
                if dist > 0.0:
                    for i in range(int(dist)):
                        pos = (int(pos1[0] + (pos2[0] - pos1[0]) * i / dist), int(pos1[1] + (pos2[1] - pos1[1]) * i / dist))
                        if pos[0] < 0 or pos[0] >= self.observe_bitmap_dims[0] or pos[1] < 0 or pos[1] >= self.observe_bitmap_dims[1]:
                            continue
                        bitmap[pos[0], pos[1], self.OBSERVATION_CHANNELS.GRAPH] = -2.0

            # Add vertices to the graph channel.
            for v in vertices:
                pos = _normPosition(self.pg.getNodePosition(v))
                if pos[0] < 0 or pos[0] >= self.observe_bitmap_dims[0] or pos[1] < 0 or pos[1] >= self.observe_bitmap_dims[1]:
                    continue
                bitmap[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.GRAPH] = v

            obs = bitmap

        # Add bitmap2 observation. This variant uses -1 to indicate unobserved nodes and agents, rather than cropping the bitmap.
        if observe_method in ["bitmap2"]:
            # Create an image which defaults to -1.
            bitmap = -1.0 * np.ones(self.observation_space(agent).shape, dtype=np.float32)

            # Set the observing agent's ID in the (0, 0) position. This is a bit hacky.
            bitmap[0, 0, self.OBSERVATION_CHANNELS.AGENT_ID] = agent.id

            def _normPosition(pos):
                x = self._minMaxNormalize(pos[0], a=0.0, b=self.observe_bitmap_dims[0], minimum=0.0, maximum=self.pg.widthPixels, eps=0.01)
                y = self._minMaxNormalize(pos[1], a=0.0, b=self.observe_bitmap_dims[1], minimum=0.0, maximum=self.pg.heightPixels, eps=0.01)
                return x, y

            # Add agents to the observation.
            for a in agents:
                pos = _normPosition(a.position)
                bitmap[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.AGENT_ID] = a.id
            
            # Add vertex idleness times to the observation.
            for v in vertices:
                pos = _normPosition(self.pg.getNodePosition(v))
                bitmap[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.IDLENESS] = self.pg.getNodeIdlenessTime(v, self.step_count)
            
            # Add edges to the graph channel.
            for edge in self.pg.graph.edges:
                pos1 = _normPosition(self.pg.getNodePosition(edge[0]))
                pos2 = _normPosition(self.pg.getNodePosition(edge[1]))
                dist = self._dist(pos1, pos2)
                if dist > 0.0:
                    for i in range(int(dist)):
                        pos = (int(pos1[0] + (pos2[0] - pos1[0]) * i / dist), int(pos1[1] + (pos2[1] - pos1[1]) * i / dist))
                        bitmap[pos[0], pos[1], self.OBSERVATION_CHANNELS.GRAPH] = -2.0

            # Add vertices to the graph channel.
            for v in self.pg.graph.nodes:
                pos = _normPosition(self.pg.getNodePosition(v))
                bitmap[int(pos[0]), int(pos[1]), self.OBSERVATION_CHANNELS.GRAPH] = v

            obs = bitmap

        # Add adjacency matrix.
        if observe_method in ["ajg_newer"]:
            # Create adjacency matrix.
            adjacency = -1.0 * np.ones((self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes()), dtype=np.float32)
            for edge in self.pg.graph.edges:
                adjacency[edge[0], edge[1]] = 1.0
                adjacency[edge[1], edge[0]] = 1.0
            obs["adjacency"] = adjacency

        # Add weighted adjacency matrix (normalized).
        if observe_method in ["adjacency"]:
            # Create adjacency matrix.
            adjacency = -1.0 * np.ones((self.pg.graph.number_of_nodes(), self.pg.graph.number_of_nodes()), dtype=np.float32)
            for edge in self.pg.graph.edges:
                maxWeight = max([self.pg.graph.edges[e]["weight"] for e in self.pg.graph.edges])
                minWeight = min([self.pg.graph.edges[e]["weight"] for e in self.pg.graph.edges])
                weight = self._minMaxNormalize(self.pg.graph.edges[edge]["weight"], minimum=minWeight, maximum=maxWeight)
                adjacency[edge[0], edge[1]] = weight
                adjacency[edge[1], edge[0]] = weight
            obs["adjacency"] = adjacency
        
        # Add agent graph position vector.
        if observe_method in ["adjacency", "ajg_newer"]:
            graphPos = {}
            # Set default value of -1.0
            for a in self.possible_agents:
                graphPos[a] = -1.0 * np.ones(3, dtype=np.float32)
            
            # Fill in actual values for agents we can see.
            for a in agents:
                vec = np.zeros(3, dtype=np.float32)
                if a.edge == None:
                    vec[0] = a.lastNode
                    vec[1] = a.lastNode
                    vec[2] = 1.0
                else:
                    vec[0] = a.edge[0]
                    vec[1] = a.edge[1]
                    vec[2] = self._getAgentPathLength(a, self._getPathToNode(a, a.edge[0])) / self.pg.graph.edges[a.edge]["weight"]
                graphPos[a] = vec
            obs["agent_graph_position"] = graphPos

        if observe_method in ["pyg"]:
            # Copy pg map to g
            g = deepcopy(self.pg.graph)
            
            # Get a list of last visit times for each node.
            lastVisits = nx.get_node_attributes(g, 'visitTime')
            
            # Get min and max idleness times for normalization.
            maxIdleness = self.step_count - min(lastVisits.values())
            minIdleness = self.step_count - max(lastVisits.values())
            allSame = maxIdleness == minIdleness

            for node in g.nodes:
                # Add normalized node idleness times as attributes in g.
                if allSame:
                    g.nodes[node]["idlenessTime"] = 1.0
                else:
                    g.nodes[node]["idlenessTime"] = self._minMaxNormalize(
                        self.step_count - lastVisits[node],
                        minimum = minIdleness,
                        maximum = maxIdleness
                    )
                
                # Add dummy lastNode and currentAction values as attributes in g.
                g.nodes[node]["lastNode"] = -1.0
                g.nodes[node]["currentAction"] = -1.0

            # Ensure that we add a node for the current agent, even if it's dead.
            if agent not in agents:
                agentsPlusEgo = agents + [agent]
            else:
                agentsPlusEgo = agents

            # Traverse through all visible agents and add their positions as new nodes to g
            for a in agentsPlusEgo:
                # To avoid node ID conflicts, generate a unique node ID
                agent_node_id = f"agent_{a.id}_pos"
                g.add_node(
                    agent_node_id,
                    pos = a.position,
                    id = -1 - a.id,
                    # id = -1.0 if a == agent else -2.0,
                    nodeType = 1,
                    visitTime = 0.0,
                    idlenessTime = 0.0,
                    lastNode = g.nodes[a.lastNode]["id"] if a.lastNode in g.nodes else -1.0,
                    currentAction = a.currentAction if a in agents else -1.0
                )

                # Check if the agent has an edge that it is currently on
                if a.edge is None:
                    # If the agent is not on an edge, add an edge from the agent's node to the node it is currently on
                    g.add_edge(agent_node_id, a.lastNode, weight=0.0)

                    # Add all of a.lastNode's neighbors as edges to the agent's node.
                    for neighbor in g.neighbors(a.lastNode):
                        if g.nodes[neighbor]["nodeType"] == 0:
                            g.add_edge(agent_node_id, neighbor, weight=g.edges[(a.lastNode, neighbor)]["weight"])
                else:
                    node1_id, node2_id = a.edge

                    # Calculate weights or set them on a case-by-case basis
                    weight_to_node1 = self._calculateEdgeWeight(a.position, g.nodes[node1_id]['pos'])
                    weight_to_node2 = self._calculateEdgeWeight(a.position, g.nodes[node2_id]['pos'])

                    g.add_edge(agent_node_id, node1_id, weight=weight_to_node1)
                    g.add_edge(agent_node_id, node2_id, weight=weight_to_node2)
            
            # Normalize the edge weights of g.
            weights = nx.get_edge_attributes(g, 'weight')
            maxWeight = max(weights.values())
            minWeight = min(weights.values())
            for edge in g.edges:
                g.edges[edge]["weight"] = self._minMaxNormalize(weights[edge], minimum=minWeight, maximum=maxWeight)
            
            # Turn g into a digraph, dg
            dg = nx.DiGraph(g)

            # Add neighbor indices to the edges.
            if self.action_method == "neighbors":
                for i in dg.nodes:
                    idx = 0
                    for j in dg.neighbors(i):
                        if dg.nodes[j]["nodeType"] == 0:
                            dg.edges[(i, j)]["neighborIndex"] = idx
                            idx += 1
                        else:
                            dg.edges[(i, j)]["neighborIndex"] = -1

            # Trim the graph to only include the nodes and edges that are visible to the agent.
            # subgraphNodes = vertices + [f"agent_{a.id}_pos" for a in agents]
            # subgraph = nx.subgraph(g, subgraphNodes)
            subgraph = dg
            subgraphNodes = list(g.nodes)

            if self.action_method == "neighbors":
                edge_attrs = ["weight", "neighborIndex"]
                # node_attrs = ["nodeType", "idlenessTime"]
                node_attrs = ["id", "nodeType", "idlenessTime", "lastNode", "currentAction"]
            else:
                edge_attrs = ["weight"]
                node_attrs = ["id", "nodeType", "idlenessTime", "lastNode", "currentAction"]

            # Convert g to PyG
            data = from_networkx(
                subgraph,
                group_node_attrs=node_attrs,
                group_edge_attrs=edge_attrs
            )
            data.x = data.x.float()
            data.edge_attr = data.edge_attr.float()

            # Calculate the agent_mask based on the graph node ID assigned to this agent.
            idx = subgraphNodes.index(f"agent_{agent.id}_pos")
            agent_mask = np.zeros(data.num_nodes, dtype=bool)
            agent_mask[idx] = True
            data.agent_idx = idx
            data.agent_mask = agent_mask

            # Set up a numpy array to hold the observation.
            # o = np.empty((1,), dtype=object)
            # o[0] = data

            obs["graph"] = data

            # Additionally, add the last node and last action of the agent to the observation.
            # obs["lastNode"] = g.nodes[agent.lastNode]["id"] if agent.lastNode in g.nodes else -1.0
            # obs["currentAction"] = agent.currentAction
        
        if (type(obs) == dict and obs == {}) or (type(obs) != dict and len(obs) < 1):
            raise ValueError(f"Invalid observation method {self.observe_method}")
        

        # Check if type of any values in obs is a graph.
        if type(obs) == dict:
            # Ensure dictionary ordering.
            obs = dict(sorted(obs.items()))

            typeSet = set([type(v) for v in obs.values()])
            if Data in typeSet:
                # If so, we want the observation to be a single-element array of objects.
                obs = np.array(list(obs.values()), dtype=object)

        return obs
    
    def _calculateEdgeWeight(self, pos1, pos2):
        '''Calculate the weights of the edges based on the position of the two points, here simply use the Euclidean distance'''
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def step(self, action_dict={}, lastStep=False):
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
        reward_dict = {agent: 0.0 for agent in self.possible_agents}
        truncated_dict = {agent: False for agent in self.possible_agents}
        info_dict = {
            agent: {
                "ready": self.dones[agent], #if done, set ready to true. Otherwise we will have buffer size problems in MAPPO due to lack of insertion.
            } for agent in self.possible_agents
        }

        # Perform attrition.
        if len(self.agents) >= self.attrition_min_agents:
            if self.attrition_method == "random":
                if random.random() < self.attrition_random_probability:
                    random_index = random.randrange(len(self.agents))
                    attrition_agent = self.agents.pop(random_index)
                    self.dones[attrition_agent] = True
                    print(f"Agent {attrition_agent.id} has been removed from the environment at step {self.step_count}.")
            elif self.attrition_method == "fixed_time":
                if self.step_count in self.attrition_times:
                    random_index = random.randrange(len(self.agents))
                    attrition_agent = self.agents.pop(random_index)
                    self.dones[attrition_agent] = True
                    print(f"Agent {attrition_agent.id} has been removed from the environment at step {self.step_count}.")

        # Perform actions.
        for agent in self.agents:
            if agent in action_dict:
                action = action_dict[agent]

                if not np.issubdtype(type(action), np.integer):
                    raise ValueError(f"Invalid action {action} of type {type(action)} provided.")

                # Interpret the action using the "full" method.
                if self.action_method == "full":
                    if action not in self.pg.graph.nodes:
                        raise ValueError(f"Invalid action {action} for agent {agent.name}")
                    dstNode = action
                
                # Interpret the action using the "neighbors" method.
                elif self.action_method == "neighbors":
                    if agent.edge == None:
                        if action > self.pg.graph.degree(agent.lastNode):
                            raise ValueError(f"Invalid action {action} for agent {agent.name}")
                        dstNode = list(self.pg.graph.neighbors(agent.lastNode))[action]
                    else:
                        if action != agent.currentAction:
                            raise ValueError(f"Invalid action {action} for agent {agent.name}")
                        dstNode = list(self.pg.graph.neighbors(agent.lastNode))[action]
                
                else:
                    raise ValueError(f"Invalid action method {self.action_method}")
                
                # Store this as the agent's last action.
                agent.currentAction = action
                
                # Provide penalty for visiting the same node twice in a row.
                # if dstNode == agent.lastNodeVisited:
                #     reward_dict[agent] -= 1.0

                # Calculate the shortest path.
                path = self._getPathToNode(agent, dstNode)
                pathLen = self._getAgentPathLength(agent, path)
                
                # Take a step towards the next node.
                stepSize = np.random.normal(loc=agent.speed, scale=1.0)
                for nextNode in path:
                    reached, stepSize = self._moveTowardsNode(agent, nextNode, stepSize)

                    # The agent has reached the next node.
                    if reached:
                        if nextNode == dstNode or not self.requireExplicitVisit:
                            # The agent has reached its destination, visiting the node.
                            # The agent receives a reward for visiting the node.
                            r = self.onNodeVisit(nextNode, self.step_count)
                            reward_dict[agent] += r

                            agent.lastNodeVisited = nextNode
                            if nextNode == dstNode:
                                agent.currentAction = -1.0
                                info_dict[agent]["ready"] = True
            
                    # The agent has exceeded its movement budget for this step.
                    if stepSize <= 0.0:
                        break

        # Record the average idleness time at this step.
        avg = self._minMaxNormalize(self.pg.getAverageIdlenessTime(self.step_count), minimum=0.0, maximum=self.step_count)
        self.avgIdlenessTimes.append(avg)        

        # Perform observations.
        for agent in self.possible_agents:

            # 3 communicaiton models here
            # agent_observation = self.observe_with_communication(agent)
            agent_observation = self.observe(agent)
            
            # Update the observation for the agent
            obs_dict[agent] = agent_observation
        
        # Record miscellaneous information.
        info_dict["node_visits"] = self.nodeVisits
        info_dict["avg_idleness"] = self.pg.getAverageIdlenessTime(self.step_count)
        info_dict["stddev_idleness"] = self.pg.getStdDevIdlenessTime(self.step_count)
        info_dict["worst_idleness"] = self.pg.getWorstIdlenessTime(self.step_count)
        info_dict["agent_count"] = len(self.agents)

        # Check truncation conditions.
        if lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles):
            for agent in self.agents:
                # Provide an end-of-episode reward.
                if self.reward_method_terminal == "average":
                    reward_dict[agent] += self.beta * self.step_count / (self.pg.getAverageIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "worst":
                    reward_dict[agent] += self.beta * self.step_count / (self.pg.getWorstIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "stddev":
                    reward_dict[agent] += self.beta * self.step_count / (self.pg.getStdDevIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "averageAverage":
                    avg = np.average(self.avgIdlenessTimes)
                    # reward_dict[agent] += self.beta * self.step_count / (avg + 1e-8)
                    reward_dict[agent] -= self.beta * avg
                elif self.reward_method_terminal == "divNormalizedWorst":
                    reward_dict[agent] /= self._minMaxNormalize(self.pg.getWorstIdlenessTime(self.step_count), minimum=0.0, maximum=self.max_cycles)
                elif self.reward_method_terminal != "none":
                    raise ValueError(f"Invalid terminal reward method {self.reward_method_terminal}")

                info_dict[agent]["ready"] = True
            
                truncated_dict[agent] = True
            self.agents = []
        
        # Provide a reward at a fixed interval.
        elif self.reward_interval >= 0 and self.step_count % self.reward_interval == 0:
            for agent in self.agents:
                # reward_dict[agent] += self.beta * self.step_count / (self.pg.getAverageIdlenessTime(self.step_count) + 1e-8)
                reward_dict[agent] -= self.beta * self._minMaxNormalize(self.pg.getAverageIdlenessTime(self.step_count), minimum=0.0, maximum=self.step_count)

        done_dict = {agent: self.dones[agent] for agent in self.possible_agents}

        # Set available actions.
        self.available_actions = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def onNodeVisit(self, node, timeStamp):
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''
        
    def onNodeVisit(self, node, timeStamp):
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''
        
        # Record the node visit.
        self.nodeVisits[node] += 1

        # Calculate a visitation reward.
        idleness = self.pg.getNodeIdlenessTime(node, timeStamp)
        avgIdleness = self.pg.getAverageIdlenessTime(timeStamp)
        reward = self._minMaxNormalize(idleness, minimum=0.0, maximum=avgIdleness)
        reward = self.alpha * reward

        # Update the node visit time.
        self.pg.setNodeVisitTime(node, timeStamp)

        return reward


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
            # Ensure that ordering is always the same for the edge.
            agent.edge = tuple(sorted((agent.lastNode, node)))

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
        return a + (x - minimum) * (b - a) / (maximum - minimum + eps)


    def _getAvailableActions(self, agent):
        ''' Returns the available actions for the given agent. '''

        if self.action_method == "full":
            if agent.edge == None:
                # All actions available.
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                actionMap[:self.pg.graph.number_of_nodes()] = 1.0
                return actionMap
            else:
                # Only the current action available (as it is still incomplete).
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                actionMap[agent.currentAction] = 1.0
                return actionMap
        
        elif self.action_method == "neighbors":
            if agent.edge == None:
                # All neighbors of the current node are available.
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                # numNeighbors = self.pg.graph.degree(agent.lastNode) - 1 # subtract 1 since the self loop adds 2 to the degree
                numNeighbors = self.pg.graph.degree(agent.lastNode)
                actionMap[:numNeighbors] = 1.0
                return actionMap
            else:
                # Only the current action available (as it is still incomplete).
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                actionMap[agent.currentAction] = 1.0
                return actionMap
        else:
            raise ValueError(f"Invalid action method {self.action_method}")
        

    def observe_with_communication(self, agent):
        ''' Adds communicated states to the agent's observation. '''

        other_agents = [temp for temp in self.agents if temp != agent ]
        agent_observation = self.observe(agent)

        for a in other_agents:
            receive_obs = self.comms_model.canReceive(a, agent)

            if receive_obs:
                agent_observation["agent_state"][a] = a.position

        return agent_observation
