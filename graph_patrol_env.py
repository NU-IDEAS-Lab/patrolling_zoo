from pettingzoo import AECEnv
import numpy as np
import random
from gym import spaces
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.patches import Circle



class GraphPatrolEnv(AECEnv):
    def __init__(self, patrol_graph, num_agents, entities, radius = 10):
        """
        Initialize the PatrolEnv object.

        Args:
            patrol_graph (PatrolGraph): The patrol graph representing the environment.
            num_agents (int): The number of agents in the environment.

        Returns:
            None
        """
        self.pg = patrol_graph
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.possible_actions = list(self.pg.graph.nodes)
        self.rewards = dict.fromkeys(self.agents, 0)
        self.dones = dict.fromkeys(self.agents, False)
        self.agentPosition = {agent: random.choice(self.possible_actions) for agent in self.agents} 
        self.next_node = dict.fromkeys(self.agents, None) 
        self.remaining_steps = dict.fromkeys(self.agents, 0) 

        self.action_spaces = {agent: spaces.Discrete(len(self.pg.graph)) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=np.inf, shape=(len(self.pg.graph),)) for agent in self.agents}

        self.observations = {agent: [] for agent in self.agents}


        self.entities = {}

        self.radius = radius


        # Spawn entities at random positions
        for entity, quantity in entities.items():
            self.entities[entity] = []
            for _ in range(quantity):
                entity_position = random.choice(list(self.pg.graph.nodes))
                self.entities[entity].append(entity_position)

        self.step_count = 0


    def reset(self):
        self.rewards = dict.fromkeys(self.agents, 0)
        self.dones = dict.fromkeys(self.agents, False)
        self.observations = {agent: random.choice(self.possible_actions) for agent in self.agents} 
        self.next_node = dict.fromkeys(self.agents, None)
        self.remaining_steps = dict.fromkeys(self.agents, 0)
        return {agent: self.observe(agent) for agent in self.agents}
    


    def step(self, action_dict = {}):
        """
        Perform a step in the environment based on the given action dictionary.

        Args:
            action_dict (dict): A dictionary containing actions for each agent.

        Returns:
            None
        """
        self.step_count += 1
        for agent in self.agents:
            # If the agent is at a node, not transitioning
            if self.remaining_steps[agent] == 0 and agent in action_dict:
                action = action_dict[agent]
                if isinstance(self.agentPosition[agent], tuple):
                    current_position = self.agentPosition[agent][0]
                else:
                    current_position = self.agentPosition[agent]

                # If the agent is asked to go where it already is
                if action == current_position:
                    self.rewards[agent] = 0  # you could also give a reward or penalty here if you wanted
                    continue

                # Update the agent's position.
                if action in self.pg.graph.nodes:
                    path = nx.shortest_path(self.pg.graph, source=current_position, target=action, weight='weight')
                    path_len = nx.shortest_path_length(self.pg.graph, source=current_position, target=action, weight='weight')

                    # If there is a direct edge
                    if path_len == 1:
                        self.agentPosition[agent] = action
                        self.rewards[agent] = self.pg.graph.edges[current_position, action]['weight']
                    else: 
                        # The agent will start transitioning to the next node on the shortest path
                        self.next_node[agent] = path[1]
                        self.remaining_steps[agent] = self.pg.graph.edges[current_position, self.next_node[agent]]['weight']
                        self.rewards[agent] = 0 # Assuming no reward until it reaches the destination node
                        # Update observation to indicate that agent is transitioning
                        self.agentPosition[agent] = (current_position, self.next_node[agent])
                else:
                    self.rewards[agent] = 0  # the action was invalid
            elif self.remaining_steps[agent] > 0:
                # The agent is transitioning
                self.remaining_steps[agent] -= 1

                # If the agent has reached the next node in its path
                if self.remaining_steps[agent] == 0:
                    self.agentPosition[agent] = self.next_node[agent]
                    self.next_node[agent] = None

        
        self.get_observations()

                    
    def last(self):
        return self.agents
    

    def plot_world(self, figsize=(18, 12)):
        """
        Plot the world representation with agent positions.

        Args:
            figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=figsize)
        markers = ['o', '*', 's', '+', 'x', 'D', 'v', '^']  # markers for entities
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']

        pos = nx.get_node_attributes(self.pg.graph, 'pos')
        nx.draw(self.pg.graph, pos, with_labels=True, node_color='lightblue', node_size=600, font_size=10, font_color='black')
        nx.draw_networkx_edge_labels(self.pg.graph, pos, edge_labels=nx.get_edge_attributes(self.pg.graph, 'weight'), font_size=7)

        # Plot agents
        for i, agent in enumerate(self.agents):
            color = colors[0]  # Red color for agents
            marker = '.'  # Small dot for agents

            if isinstance(self.agentPosition[agent], tuple):  # the agent is transitioning
                pos1, pos2 = [self.pg.getNodePosition(node) for node in self.agentPosition[agent]]
                ratio = self.remaining_steps[agent] / self.pg.graph.edges[self.agentPosition[agent]]['weight']
                pos = (pos1[0] * ratio + pos2[0] * (1 - ratio), pos1[1] * ratio + pos2[1] * (1 - ratio))
                label = f'{agent} : {self.agentPosition[agent][0]} --> {self.agentPosition[agent][1]}'
            else:  # the agent is at a node
                pos = self.pg.getNodePosition(self.agentPosition[agent])
                label = f'{agent} : {self.agentPosition[agent]}'

            plt.scatter(*pos, color=color, marker=marker, zorder=10, alpha=0.8, s=100)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=label, alpha=0.8)

        # Plot entities
        entity_color = 'purple'  # choose a specific color for entities
        for i, (entity, positions) in enumerate(self.entities.items()):
            marker = markers[i % len(markers)]  # choose a different marker for each type of entity
            color = color[i % len(colors)]  # choose a different color for each type of entity
            for pos in positions:
                pos = self.pg.getNodePosition(pos)
                plt.scatter(*pos, color=entity_color, marker=marker, zorder=10, alpha=0.5, s=100)
            plt.plot([], [], color=entity_color, marker=marker, linestyle='None', label=entity, alpha=0.5)

        # Add scale line
        scale_line_length = self.radius
        x, y = np.array([0, scale_line_length]), np.array([0, 0])
        ax.plot(x, y+2, color='red', linestyle='dashed', linewidth=1)
       
        plt.legend()
        plt.text(0, 0, f'Current step: {self.step_count}')
        plt.show()





    def euclidean_distance(self, pos1, pos2):
        """
        Calculate Euclidean distance between two points.

        Args:
            pos1 (tuple): The first point.
            pos2 (tuple): The second point.

        Returns:
            float: The Euclidean distance between the two points.
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


    def get_observations(self):
        """
        Get the observation for each agent.

        Args:
            None

        Returns:
            None
        """

        self.observations = {agent: [] for agent in self.agents}
        
        for agent in self.agents:
            if isinstance(self.agentPosition[agent], tuple):  # the agent is transitioning
                agent_pos_key = self.agentPosition[agent]
            else:  # the agent is at a node
                agent_pos_key = self.agentPosition[agent]

            # Check all entities
            for entity, positions in self.entities.items():
                for pos in positions:
                    if isinstance(pos, tuple):  # the entity is transitioning
                        entity_pos_key = pos
                    else:  # the entity is at a node
                        entity_pos_key = pos
                    self.observations[agent].append((entity, entity_pos_key))  # store the entity name and its vertex

            # Check all other agents
            for other_agent in self.agents:
                if other_agent == agent:  # don't check the agent with itself
                    continue
                if isinstance(self.agentPosition[other_agent], tuple):  # the other agent is transitioning
                    other_agent_pos_key = self.agentPosition[other_agent]
                else:  # the other agent is at a node
                    other_agent_pos_key = self.agentPosition[other_agent]
                self.observations[agent].append((other_agent, other_agent_pos_key))  # store the other agent name and its vertex
