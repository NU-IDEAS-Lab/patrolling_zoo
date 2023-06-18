from pettingzoo import AECEnv
import numpy as np
import random
from gym import spaces
import matplotlib.pyplot as plt
import networkx as nx

from matplotlib.patches import Circle



from pettingzoo import AECEnv
from gym import spaces
import numpy as np
import random

class GraphPatrolEnv(AECEnv):
    def __init__(self, patrol_graph, num_agents, entities = {}, radius=10):
        """
        Initialize the PatrolEnv object.

        Args:
            patrol_graph (PatrolGraph): The patrol graph representing the environment.
            num_agents (int): The number of agents in the environment.

        Returns:
            None
        """
        self.graph = patrol_graph
        self.agents = [f'agent_{i}' for i in range(num_agents)]
        self.possible_actions = list(self.graph.nodes)
        #self.rewards = dict.fromkeys(self.agents, 0)
        self.reward = 0           
        self.dones = dict.fromkeys(self.agents, False)
        self.agentPosition = {agent: random.choice(self.possible_actions) for agent in self.agents} 
        self.next_node = dict.fromkeys(self.agents, None) 
        self.remaining_steps = dict.fromkeys(self.agents, 0) 

        self.idle_time = {node: 0 for node in self.graph.nodes}

        self.entities = {}
        self.radius = radius

        # Spawn entities at random positions
        for entity, quantity in entities.items():
            self.entities[entity] = []
            for _ in range(quantity):
                entity_position = random.choice(list(self.graph.nodes))
                self.entities[entity].append(entity_position)

        self.step_count = 0

    def action_space(self, agent):
        """
        Define the action space for a given agent.

        Args:
            agent: The agent for which to define the action space.

        Returns:
            gym.spaces.Space: The action space for the agent.
        """
        return spaces.Discrete(len(self.graph))

    def observation_space(self, agent):
        """
        Define the observation space for a given agent.

        Args:
            agent: The agent for which to define the observation space.

        Returns:
            gym.spaces.Space: The observation space for the agent.
        """
        return spaces.Box(low=0, high=np.inf, shape=(len(self.graph),))


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


        # The idling time of each node is increased by one
        for node in self.idle_time:
            self.idle_time[node] += 1


        for agent in self.agents:
            # If the agent is at a node, not transitioning
            if self.remaining_steps[agent] == 0 and agent in action_dict:
                action = action_dict[agent]
                current_position = self.agentPosition[agent]

                # If the agent is asked to go where it already is
                if action == current_position:
                    continue

                # Update the agent's position.
                if action in self.graph.nodes:
                    path = nx.shortest_path(self.graph, source=current_position, target=action, weight='weight')
                    path_len = nx.shortest_path_length(self.graph, source=current_position, target=action, weight='weight')

                    # If there is a direct edge
                    if path_len == 1:
                        self.agentPosition[agent] = action
                    else: 
                        # The agent will start transitioning to the next node on the shortest path
                        self.next_node[agent] = path[1]
                        self.remaining_steps[agent] = self.graph.edges[current_position, self.next_node[agent]]['weight']
                        # Update observation to indicate that agent is transitioning
                        self.agentPosition[agent] = (current_position, self.next_node[agent])
                else:
                    pass # the action was invalid
            elif self.remaining_steps[agent] > 0:
                # The agent is transitioning
                self.remaining_steps[agent] -= 1

                # If the agent has reached the next node in its path
                if self.remaining_steps[agent] == 0:
                    self.agentPosition[agent] = self.next_node[agent]
                    self.next_node[agent] = None
        
        # The idling time of nodes occupied by an agent is resetted to 0
        for agent in self.agents:
            position = self.agentPosition[agent]
            if isinstance(self.agentPosition[agent], int):
                self.idle_time[position] = 0

        self.reward = sum(self.idle_time)/len(self.idle_time)        
                    
    def last(self):
        return self.agents
    

    def render(self, figsize=(16, 9)):
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

        pos = nx.get_node_attributes(self.graph, 'pos')
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=600, font_size=10, font_color='black')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=nx.get_edge_attributes(self.graph, 'weight'), font_size=7)

        # Plot agents
        for i, agent in enumerate(self.agents):
            color = colors[0]  # Red color for agents
            marker = '.'  # Small dot for agents

            if isinstance(self.agentPosition[agent], tuple):  # the agent is transitioning
                pos1, pos2 = [self.graph.nodes[node]["pos"] for node in self.agentPosition[agent]]
                ratio = self.remaining_steps[agent] / self.graph.edges[self.agentPosition[agent]]['weight']
                pos = (pos1[0] * ratio + pos2[0] * (1 - ratio), pos1[1] * ratio + pos2[1] * (1 - ratio))
                label = f'{agent} : {self.agentPosition[agent][0]} --> {self.agentPosition[agent][1]}'
            else:  # the agent is at a node
                pos = self.graph.nodes[self.agentPosition[agent]]["pos"]
                label = f'{agent} : {self.agentPosition[agent]}'

            plt.scatter(*pos, color=color, marker=marker, zorder=20, alpha=1, s=100)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=label, alpha=0.8)

        # Plot entities
        entity_color = 'purple'  # choose a specific color for entities
        for i, (entity, positions) in enumerate(self.entities.items()):
            marker = markers[i % len(markers)]  # choose a different marker for each type of entity
            color = colors[i % len(colors)]  # choose a different color for each type of entity
            for pos in positions:
                pos = self.graph.nodes[pos]["pos"]
                plt.scatter(*pos, color=entity_color, marker=marker, zorder=10, alpha=0.3, s=200)
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


    def observe(self, agent):
        observation = {"agents": [], "entities": []}

        # Calculate the current position of the observing agent
        agent_pos = self.calc_agent_pos(agent)

        for other_agent in self.agents:
            if other_agent == agent:  # don't check the agent with itself
                continue

            # Calculate the other agent's position
            otherPos = self.calc_agent_pos(other_agent)

            if self.euclidean_distance(agent_pos, otherPos) < self.radius:
                observation["agents"].append((other_agent, self.agentPosition[other_agent]))

        # Same for entities
        for entity, positions in self.entities.items():
            for entity_position in positions:
                entityPos = self.graph.nodes[entity_position]["pos"]
                if self.euclidean_distance(agent_pos, entityPos) < self.radius:
                    observation["entities"].append((entity, entity_position))

        return observation


    def calc_agent_pos(self, agent):
        """Calculates the position of an agent, even if it's transitioning between nodes.

        Args:
            agent (str): The agent to calculate the position for.

        Returns:
            tuple: The (x, y) position of the agent.
        """
        if isinstance(self.agentPosition[agent], tuple):  # the agent is transitioning
            pos1, pos2 = [self.graph.nodes[node]["pos"] for node in self.agentPosition[agent]]
            ratio = self.remaining_steps[agent] / self.graph.edges[self.agentPosition[agent]]['weight']
            agent_pos = (pos1[0] * ratio + pos2[0] * (1 - ratio), pos1[1] * ratio + pos2[1] * (1 - ratio))
        else:  # the agent is at a node
            agent_pos = self.graph.nodes[self.agentPosition[agent]]["pos"]

        return agent_pos