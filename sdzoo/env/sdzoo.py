from pettingzoo.utils.env import ParallelEnv
from sdzoo.env.communication_model import CommunicationModel
from sdzoo.env.sd_graph import NODE_TYPE
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


class SDAgent():
    ''' This class stores all agent state. '''

    def __init__(self, id, position=(0.0, 0.0), speed = 1.0, observationRadius=np.inf, startingNode=None, currentState = 1, max_nodes = 50, max_capacity = 1):
        self.id = id
        self.name = f"agent_{id}"
        self.startingPosition = position
        self.startingSpeed = speed
        self.startingNode = startingNode
        self.observationRadius = observationRadius
        self.currentState = currentState
        self.max_nodes = max_nodes
        self.max_capacity = max_capacity
        self.reset()
    
    
    def reset(self):
        self.position = self.startingPosition
        self.speed = self.startingSpeed
        self.edge = None
        self.currentAction = -1.0
        self.lastNode = self.startingNode
        self.lastNodeVisited = None
        self.stateBelief = {i: -1.0 for i in range(self.max_nodes)}
        self.payloads = 0
     

class parallel_env(ParallelEnv): 
    metadata = {
        "name": "sdzoo_environment_v0",
    }

    def __init__(self, sd_graph, num_agents,
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
                 max_nodes: int = 50,
                 max_neighbors: int = 15,
                 reward_interval: int = -1,
                 regenerate_graph_on_reset: bool = False,
                 *args,
                 **kwargs):
        """
        Initialize the PatrolEnv object.

        Args:
            sd_graph (SDGraph): The patrol graph representing the environment.
            num_agents (int): The number of agents in the environment.

        Returns:
            None
        """
        super().__init__(*args, **kwargs)

        self.sdg = sd_graph

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
        self.max_nodes = max_nodes
        self.max_neighbors = max_neighbors

        self.reward_interval = reward_interval

        self.alpha = alpha
        self.beta = beta

        # Create the agents with random starting positions.
        self.agentOrigins = random.sample(list(self.sdg.graph.nodes), num_agents)
        startingPositions = [self.sdg.getNodePosition(i) for i in self.agentOrigins]
        self.possible_agents = [
            SDAgent(i, startingPositions[i],
                        speed = speed,
                        startingNode = self.agentOrigins[i],
                        observationRadius = self.observationRadius,
                        max_nodes = self.max_nodes
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
            if self.action_full_max_nodes < len(self.sdg.graph):
                raise ValueError("The action space is smaller than the graph size.")
            maxNodes = self.action_full_max_nodes if self.action_full_max_nodes > 0 else len(self.sdg.graph)
            return spaces.Discrete(maxNodes + 2) # add 2 for load and drop
        
        elif action_method == "neighbors":
            maxDegree = self.action_neighbors_max_degree # just use a fixed size and mask it
            return spaces.Discrete(maxDegree + 2) # add 2 for load and drop


    def _buildStateSpace(self, observe_method):
        ''' Creates a state space given the observation method.
            Returns a gym.spaces.* object. '''
        
        # Create the state space dictionary.
        state_space = {}

        # Add to the dictionary depending on the observation method.

        # Add agent id.
        if observe_method in ["ajg_new", "ajg_newer", "adjacency"]:
            state_space["agent_id"] = spaces.Box(
                low = -1,
                high = len(self.possible_agents),
                dtype=np.int32
            )

        # Add vertex people/payload information 
        if observe_method in ["ranking", "raw", "old", "ajg_new", "ajg_newer", "adjacency", "idlenessOnly"]:
            state_space["vertex_state"] = spaces.Dict({
                v: spaces.Box(
                    # people, payloads
                    low = np.array([-1.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf], dtype=np.float32)
                ) for v in range(self.sdg.graph.number_of_nodes())
            }) # type: ignore
        
        # Add vertex distances from each agent.
        if observe_method in ["old", "ajg_new", "ajg_newer"]:
            state_space["vertex_distances"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([0.0] * self.sdg.graph.number_of_nodes(), dtype=np.float32),
                    high = np.array([np.inf] * self.sdg.graph.number_of_nodes(), dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        # Add adjacency matrix.
        if observe_method in ["adjacency", "ajg_newer"]:
            state_space["adjacency"] = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(self.sdg.graph.number_of_nodes(), self.sdg.graph.number_of_nodes()),
                dtype=np.float32,
            )
        
        # Add agent graph position vector.
        if observe_method in ["adjacency", "ajg_newer"]:
            state_space["agent_graph_position"] = spaces.Dict({
                a: spaces.Box(
                    low = np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                    high = np.array([self.sdg.graph.number_of_nodes(), self.sdg.graph.number_of_nodes(), 1.0], dtype=np.float32),
                ) for a in self.possible_agents
            }) # type: ignore
        
        if observe_method in ["pyg"]:
            if self.action_method == "neighbors":
                edge_space = spaces.Box(
                    # weight, neighborID
                    low = np.array([0.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf], dtype=np.float32),
                )
                node_space = spaces.Box(
                    # nodeType, idleness, degree, people, payloads
                    low = np.array([-np.inf, -1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                )
                node_type_idx = 0
            else:
                edge_space = spaces.Box(
                    # weight
                    low = np.array([0.0], dtype=np.float32),
                    high = np.array([np.inf], dtype=np.float32),
                )
                node_space = spaces.Box(
                    # ID, nodeType, idleness, lastNode, currentAction
                    low = np.array([0.0, -np.inf, -1.0, -1.0, -1.0], dtype=np.float32),
                    high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
                )
                node_type_idx = 1

            state_space["graph"] = spaces.Graph(
                node_space = node_space,
                edge_space = edge_space
            )
            state_space["graph"].node_type_idx = node_type_idx
        
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
        self.sdg.reset(seed, randomizeIds=randomizeIds, regenerateGraph=regenerateGraph)

        # Reset the information about idleness over time.
        self.avgIdlenessTimes = []

        # Reset the node visit counts.
        self.nodeVisits = np.zeros(self.sdg.graph.number_of_nodes())

        # Reset the agents.
        self.agentOrigins = random.sample(list(self.sdg.graph.nodes), len(self.possible_agents))
        startingPositions = [self.sdg.getNodePosition(i) for i in self.agentOrigins]
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
        pos = nx.get_node_attributes(self.sdg.graph, 'pos')
        idleness = [self.sdg.getNodeIdlenessTime(i, self.step_count) for i in self.sdg.graph.nodes]
        nodeColors = [self._minMaxNormalize(idleness[i], a=0.0, b=100, minimum=0.0, maximum=self.step_count) for i in self.sdg.graph.nodes]
        nx.draw_networkx(self.sdg.graph,
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
        weights = {key: np.round(value, 1) for key, value in nx.get_edge_attributes(self.sdg.graph, 'weight').items()}
        nx.draw_networkx_edge_labels(self.sdg.graph, pos, edge_labels=weights, font_size=7)
        
        # Draw the agents.
        for i, agent in enumerate(self.possible_agents):
            marker = markers[i % len(markers)] if agent in self.agents else markers_done[i % len(markers_done)]
            color = colors[i % len(colors)]
            plt.scatter(*agent.position, color=color, marker=marker, zorder=10, alpha=0.3, s=300)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=agent.name, alpha=0.5)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.gcf().text(0,0,f'Current step: {self.step_count}, Average idleness time: {self.sdg.getAverageIdlenessTime(self.step_count):.2f}')
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


    def communicate(self, receiver=None, allow_done_agents=False):
        ''' Simulates messages sent by all agents to receiver.
            If received successfully, the receiver updates its state belief.
            If receiver is set to none, will perform communication for all. '''

        # Determine which agents can communicate.
        if allow_done_agents:
            agentList = self.possible_agents
        else:
            agentList = self.agents
        
        if receiver == None:
            receiverList = agentList
        else:
            receiverList = [receiver]

        # Perform communication.
        for rcvr in receiverList:
            for a in agentList:
                if a != rcvr and self.comms_model.canReceive(a, rcvr):
                    for v in self.sdg.graph.nodes:
                        if self._dist(self.sdg.getNodePosition(v), a.position) <= a.observationRadius:
                            rcvr.stateBelief[v] = (self.sdg.getNodePayloads(v), self.sdg.getNodePeople(v))


    def _populateStateSpace(self, observe_method, agent, radius, allow_done_agents): # TODO: add payloads
        ''' Returns a populated state/observation space.'''

        if radius == None:
            radius = agent.observationRadius

        if allow_done_agents:
            agentList = self.possible_agents
        else:
            agentList = self.agents

        # Calculate the list of visible agents and vertices.
        vertices = [v for v in self.sdg.graph.nodes if self._dist(self.sdg.getNodePosition(v), agent.position) <= radius]
        agents = [a for a in agentList if self._dist(a.position, agent.position) <= radius]

        # Update beliefs for nodes which we can see.
        for v in vertices:
            agent.stateBelief[v] = (self.sdg.getNodePayloads(v), self.sdg.getNodePeople(v))

        # Perform communication.
        for a in agentList:
            if a != agent and self.comms_model.canReceive(a, agent):
                if a not in agents:
                    agents.append(a)
                for v in self.sdg.graph.nodes:
                    if self._dist(self.sdg.getNodePosition(v), a.position) <= radius:
                        if v not in vertices:
                            vertices.append(v)
                        # Update state belief for communicates nodes.
                        agent.stateBelief[v] = (self.sdg.getNodePayloads(v), self.sdg.getNodePeople(v))
        
        agents = sorted(agents, key=lambda a: a.id)
        vertices = sorted(vertices)
        
        obs = {}

        # Add agent ID.
        # if observe_method in ["ajg_new", "ajg_newer", "adjacency", "pyg"]:
        if observe_method in ["ajg_new", "ajg_newer", "adjacency"]:
            obs["agent_id"] = agent.id

        # Add vertex idleness time (minMax normalized).
        if observe_method in ["ajg_new", "ajg_newer"]:
            # Create numpy array of idleness times.
            idlenessTimes = np.zeros(self.sdg.graph.number_of_nodes())
            for v in vertices:
                idlenessTimes[v] = self.sdg.getNodeIdlenessTime(v, self.step_count)
            
            # Normalize.
            if np.size(idlenessTimes) > 0:
                if np.min(idlenessTimes) == np.max(idlenessTimes):
                    idlenessTimes = np.ones(self.sdg.graph.number_of_nodes())
                else:
                    idlenessTimes = self._minMaxNormalize(idlenessTimes)

            # Create dictionary with default value of -1.0.
            obs["vertex_state"] = {v: -1.0 for v in range(self.sdg.graph.number_of_nodes())}

            # Fill actual values for nodes we can see.
            for v in vertices:
                obs["vertex_state"][v] = idlenessTimes[v]

        # Add vertex idleness time (raw).
        if observe_method in ["raw", "old", "idlenessOnly", "adjacency"]:
            # Create dictionary with default value of -1.0.
            obs["vertex_state"] = {v: -1.0 for v in range(self.sdg.graph.number_of_nodes())}

            for node in vertices:
                obs["vertex_state"][node] = self.sdg.getNodeIdlenessTime(node, self.step_count)

        # Add vertex distances from each agent (normalized).
        if observe_method in ["ajg_new", "ajg_newer"]:
            # Calculate the shortest path distances from each agent to each node.
            vDists = np.ones((len(self.possible_agents), self.sdg.graph.number_of_nodes()))
            for a in agents:
                for v in self.sdg.graph.nodes:
                    path = self._getPathToNode(a, v)
                    dist = self._getAgentPathLength(a, path)
                    dist = self._minMaxNormalize(dist, minimum=0.0, maximum=self.sdg.longestPathLength)
                    vDists[a.id, v] = dist
            
            # Convert to dictionary.
            vertexDistances = {}
            for a in self.possible_agents:
                vertexDistances[a] = vDists[a.id]
            
            obs["vertex_distances"] = vertexDistances

        # Add weighted adjacency matrix (normalized).
        if observe_method in ["adjacency"]:
            # Create adjacency matrix.
            adjacency = -1.0 * np.ones((self.sdg.graph.number_of_nodes(), self.sdg.graph.number_of_nodes()), dtype=np.float32)
            for edge in self.sdg.graph.edges:
                maxWeight = max([self.sdg.graph.edges[e]["weight"] for e in self.sdg.graph.edges])
                minWeight = min([self.sdg.graph.edges[e]["weight"] for e in self.sdg.graph.edges])
                weight = self._minMaxNormalize(self.sdg.graph.edges[edge]["weight"], minimum=minWeight, maximum=maxWeight)
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
                    vec[2] = self._getAgentPathLength(a, self._getPathToNode(a, a.edge[0])) / self.sdg.graph.edges[a.edge]["weight"]
                graphPos[a] = vec
            obs["agent_graph_position"] = graphPos

        if observe_method in ["pyg"]:
            # Copy pg map to g
            g = deepcopy(self.sdg.graph)
 
            # Get a list of last visit times for each node.
            lastVisits = [agent.stateBelief[i] for i in range(self.sdg.graph.number_of_nodes())]
            
            # Get min and max idleness times for normalization.
            maxIdleness = self.step_count - min(lastVisits)
            minIdleness = self.step_count - max(lastVisits)
            allSame = maxIdleness == minIdleness

            # Set attributes of patrol graph nodes.
            for node in g.nodes:
                # Add dummy lastNode and currentAction values as attributes in g for all nodes.
                g.nodes[node]["lastNode"] = -1.0
                g.nodes[node]["currentAction"] = -1.0

                # Normalize idleness times for visible nodes
                if allSame:
                    g.nodes[node]["idlenessTime"] = 1.0
                else:
                    g.nodes[node]["idlenessTime"] = self._minMaxNormalize(
                        self.step_count - lastVisits[node],
                        minimum=minIdleness,
                        maximum=maxIdleness
                    )
                        
                if node in vertices:
                    # Node is visible.
                    g.nodes[node]["nodeType"] = NODE_TYPE.OBSERVABLE_NODE
                else:
                    # Node is not visible.
                    g.nodes[node]["nodeType"] = NODE_TYPE.UNOBSERVABLE_NODE

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
                    nodeType = NODE_TYPE.AGENT,
                    visitTime = 0.0,
                    idlenessTime = 0.0,
                    payloads = a.payloads,
                    max_capacity = a.max_capacity,
                    lastNode = g.nodes[a.lastNode]["id"] if a.lastNode in g.nodes else -1.0,
                    currentAction = a.currentAction if a in agents else -1.0
                )

                # Check if the agent has an edge that it is currently on
                if a.edge is None:
                    # If the agent is not on an edge, add an edge from the agent's node to the node it is currently on
                    g.add_edge(agent_node_id, a.lastNode, weight=0.0)

                    # Add all of a.lastNode's neighbors as edges to the agent's node.
                    for neighbor in g.neighbors(a.lastNode):
                        if g.nodes[neighbor]["nodeType"] != NODE_TYPE.AGENT:
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

            if self.action_method == "neighbors":
                for i in dg.nodes:
                    # Add degree to the node features.
                    dg.nodes[i]["degree"] = dg.out_degree(i)

                    # Add neighbor indices to the edges.
                    idx = 0
                    for j in dg.neighbors(i):
                        if dg.nodes[j]["nodeType"] == NODE_TYPE.AGENT:
                            dg.edges[(i, j)]["neighborIndex"] = -1
                        else:
                            dg.edges[(i, j)]["neighborIndex"] = idx
                            idx += 1

            # Trim the graph to only include the nodes and edges that are visible to the agent.
            # subgraphNodes = vertices + [f"agent_{a.id}_pos" for a in agents]
            # subgraph = nx.subgraph(g, subgraphNodes)
            subgraph = dg
            subgraphNodes = list(g.nodes)

            if self.action_method == "neighbors":
                edge_attrs = ["weight", "neighborIndex"]
                node_attrs = ["nodeType", "idlenessTime", "degree", "people", "payloads"]
                # node_attrs = ["id", "nodeType", "idlenessTime", "lastNode", "currentAction"]
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
            if agent.edge == None:
                idx = subgraphNodes.index(agent.lastNode)
                neighborhood = list(subgraph.neighbors(agent.lastNode))
            else:
                idx = subgraphNodes.index(f"agent_{agent.id}_pos")
                neighborhood = list(subgraph.neighbors(f"agent_{agent.id}_pos"))
            agent_mask = np.zeros(data.num_nodes, dtype=bool)
            agent_mask[idx] = True
            data.agent_idx = idx
            data.agent_mask = agent_mask

            # Calculate neighbor information.
            neighbors = []
            for neighbor in neighborhood:
                if subgraph.nodes[neighbor]["nodeType"] != NODE_TYPE.AGENT:
                    neighbors.append(subgraphNodes.index(neighbor))
            nbrMask = np.zeros(self.max_nodes, dtype=bool)
            nbrMask[neighbors] = True
            data.neighbors = neighbors
            data.neighbors_mask = nbrMask

            obs["graph"] = data


        if (type(obs) == dict and obs == {}) or (type(obs) != dict and len(obs) < 1):
            raise ValueError(f"Invalid observation method {self.observe_method}")
        

        # Check if type of any values in obs is a graph.
        if type(obs) == dict:
            # Ensure dictionary ordering.
            obs = dict(sorted(obs.items()))

            typeSet = set([type(v) for v in obs.values()])
            if Data in typeSet:
                # If so, we want the observation to be a single-element array of objects.
                o = np.empty((len(obs),), dtype=object)
                for i, k in enumerate(obs.keys()):
                    o[i] = obs[k]
                obs = o

        return obs
    
    def _calculateEdgeWeight(self, pos1, pos2):
        '''Calculate the weights of the edges based on the position of the two points, here simply use the Euclidean distance'''
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def step(self, action_dict={}, lastStep=False): # TODO: add load and drop steps for agents, use negative numbers for load and drop
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
        reward_dict = {agent: 0.0 for agent in self.possible_agents} # TODO: add positive rewards for dropping correct loads, picking up correct loads, negative rewards for picking up/dropping off incorrect loads
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
        for agent in self.agents: # TODO: add loading/dropping off payloads
            if agent in action_dict:
                action = action_dict[agent]

                if not np.issubdtype(type(action), np.integer):
                    raise ValueError(f"Invalid action {action} of type {type(action)} provided.")

                # Get the destination node.
                dstNode = self.getDestinationNode(agent, action)

                # Store this as the agent's last action.
                agent.currentAction = action
                
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
                        # Agent reached the destination, assign a new speed from normal distribution
                        # agent.speed = max(np.random.normal(loc=agent.startingSpeed, scale=5.0), 1.0)
            
                    # The agent has exceeded its movement budget for this step.
                    if stepSize <= 0.0:
                        break

        # Record the average idleness time at this step.
        avg = self._minMaxNormalize(self.sdg.getAverageIdlenessTime(self.step_count), minimum=0.0, maximum=self.step_count)
        self.avgIdlenessTimes.append(avg)        

        # Perform observations.
        for agent in self.possible_agents:
            agent_observation = self.observe(agent)
            obs_dict[agent] = agent_observation
        
        # Record miscellaneous information.
        info_dict["node_visits"] = self.nodeVisits
        info_dict["avg_idleness"] = self.sdg.getAverageIdlenessTime(self.step_count)
        info_dict["stddev_idleness"] = self.sdg.getStdDevIdlenessTime(self.step_count)
        info_dict["worst_idleness"] = self.sdg.getWorstIdlenessTime(self.step_count)
        info_dict["agent_count"] = len(self.agents)

        # Check truncation conditions.
        if lastStep or (self.max_cycles >= 0 and self.step_count >= self.max_cycles):
            for agent in self.agents:
                # Provide an end-of-episode reward.
                if self.reward_method_terminal == "average":
                    reward_dict[agent] += self.beta * self.step_count / (self.sdg.getAverageIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "worst":
                    reward_dict[agent] += self.beta * self.step_count / (self.sdg.getWorstIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "stddev":
                    reward_dict[agent] += self.beta * self.step_count / (self.sdg.getStdDevIdlenessTime(self.step_count) + 1e-8)
                elif self.reward_method_terminal == "averageAverage":
                    avg = np.average(self.avgIdlenessTimes)
                    # reward_dict[agent] += self.beta * self.step_count / (avg + 1e-8)
                    reward_dict[agent] -= self.beta * avg
                elif self.reward_method_terminal == "divNormalizedWorst":
                    reward_dict[agent] /= self._minMaxNormalize(self.sdg.getWorstIdlenessTime(self.step_count), minimum=0.0, maximum=self.max_cycles)
                elif self.reward_method_terminal != "none":
                    raise ValueError(f"Invalid terminal reward method {self.reward_method_terminal}")

                info_dict[agent]["ready"] = True
            
                truncated_dict[agent] = True
            self.agents = []
        
        # Provide a reward at a fixed interval.
        elif self.reward_interval >= 0 and self.step_count % self.reward_interval == 0:
            for agent in self.agents:
                # reward_dict[agent] += self.beta * self.step_count / (self.sdg.getAverageIdlenessTime(self.step_count) + 1e-8)
                reward_dict[agent] -= self.beta * self._minMaxNormalize(self.sdg.getAverageIdlenessTime(self.step_count), minimum=0.0, maximum=self.step_count)

        done_dict = {agent: self.dones[agent] for agent in self.possible_agents}

        # Set available actions.
        self.available_actions = {agent: self._getAvailableActions(agent) for agent in self.possible_agents}

        return obs_dict, reward_dict, done_dict, truncated_dict, info_dict


    def onNodeVisit(self, node, timeStamp): # TODO: change for dropping/picking up loads
        ''' Called when an agent visits a node.
            Returns the reward for visiting the node, which is proportional to
            node idleness time. '''
        
        # Record the node visit.
        self.nodeVisits[node] += 1

        # Calculate a visitation reward.
        idleness = self.sdg.getNodeIdlenessTime(node, timeStamp)
        avgIdleness = self.sdg.getAverageIdlenessTime(timeStamp)
        reward = self._minMaxNormalize(idleness, minimum=0.0, maximum=avgIdleness)
        reward = self.alpha * reward

        # Update the node visit time.
        self.sdg.setNodeVisitTime(node, timeStamp)

        return reward


    def getDestinationNode(self, agent, action): # TODO: inspect for changes needed regarding loading/dropping payloads
        ''' Returns the destination node for the given agent and action. '''

        # Interpret the action using the "full" method.
        if self.action_method == "full":
            if action not in self.sdg.graph.nodes:
                raise ValueError(f"Invalid action {action} for agent {agent.name}")
            dstNode = action
        
        # Interpret the action using the "neighbors" method.
        elif self.action_method == "neighbors":
            if agent.edge == None:
                if action >= self.sdg.graph.degree(agent.lastNode):
                    raise ValueError(f"Invalid action {action} for agent {agent.name}. Node {agent.lastNode} has only {self.sdg.graph.degree(agent.lastNode)} neighbors.")
                dstNode = list(self.sdg.graph.neighbors(agent.lastNode))[action]
            else:
                if action != agent.currentAction:
                    raise ValueError(f"Invalid action {action} for agent {agent.name}. Must complete action {agent.currentAction} first.")
                dstNode = list(self.sdg.graph.neighbors(agent.lastNode))[action]
        
        else:
            raise ValueError(f"Invalid action method {self.action_method}")
        
        return dstNode


    def _moveTowardsNode(self, agent, node, stepSize): # TODO: inspect for changes needed regarding loading/dropping payloads
        ''' Takes a single step towards the next node.
            Returns a tuple containing whether the agent has reached the node
            and the remaining step size. '''

        # Take a step towards the next node.
        posNextNode = self.sdg.getNodePosition(node)
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
            path1 = nx.shortest_path(self.sdg.graph, source=agent.edge[0], target=dstNode, weight='weight')
            pathLen1 = self._getAgentPathLength(agent, path1)
            path2 = nx.shortest_path(self.sdg.graph, source=agent.edge[1], target=dstNode, weight='weight')
            pathLen2 = self._getAgentPathLength(agent, path2)
            path = path1
            if pathLen2 < pathLen1:
                path = path2
        
        # The agent is on a node. Simply calculate the shortest path.
        else:
            path = nx.shortest_path(self.sdg.graph, source=agent.lastNode, target=dstNode, weight='weight')

            # Remove the first node from the path if the destination is different than the current node.
            if agent.lastNode != dstNode:
                path = path[1:]
        
        return path


    def _getAgentPathLength(self, agent, path):
        ''' Calculates the length of the given path for the given agent. '''

        pathLen = 0.0
        pathLen += self._dist(agent.position, self.sdg.getNodePosition(path[0]))
        pathLen += nx.path_weight(self.sdg.graph, path, weight='weight')

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
                actionMap[:self.sdg.graph.number_of_nodes()] = 1.0
                return actionMap
            else:
                # Only the current action available (as it is still incomplete).
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                actionMap[agent.currentAction] = 1.0
                return actionMap
        
        elif self.action_method == "neighbors":
            if agent.edge == None: # TODO: add load/drop when not on edge
                # All neighbors of the current node are available.
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                # numNeighbors = self.sdg.graph.degree(agent.lastNode) - 1 # subtract 1 since the self loop adds 2 to the degree
                numNeighbors = self.sdg.graph.degree(agent.lastNode)
                actionMap[:numNeighbors] = 1.0
                return actionMap
            else:
                # Only the current action available (as it is still incomplete).
                actionMap = np.zeros(self.action_space(agent).n, dtype=np.float32)
                actionMap[agent.currentAction] = 1.0
                return actionMap
        else:
            raise ValueError(f"Invalid action method {self.action_method}")