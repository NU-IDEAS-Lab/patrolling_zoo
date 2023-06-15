from pettingzoo.utils.env import ParallelEnv
from gymnasium import spaces
import random
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx


class PatrolAgentState():
    def __init__(self, position=(0.0, 0.0)):
        self.position = position

class PatrolVertexState():
    def __init__(self, idlenessTime=0.0):
        self.idlenessTime = idlenessTime

class PatrollingZooEnvironment(ParallelEnv):
    metadata = {
        "name": "patrolling_zoo_environment_v0",
    }

    def __init__(self, patrol_graph, num_agents):
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
        self.action_spaces = {agent: spaces.Discrete(len(self.pg.graph)) for agent in self.agents}
        self.observation_spaces = {agent: spaces.Box(low=0, high=np.inf, shape=(len(self.pg.graph),)) for agent in self.agents}

        self.reset()

        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.rewards = dict.fromkeys(self.agents, 0)
        self.dones = dict.fromkeys(self.agents, False)
        self.observations = dict.fromkeys(self.agents, 0)
        self.next_node = dict.fromkeys(self.agents, None)
        self.remaining_steps = dict.fromkeys(self.agents, 0)
        return {agent: self.observe(agent) for agent in self.agents}

    def render(self, figsize=(18, 12)):
        """
        Plot the world representation with agent positions.

        Args:
            figsize (tuple, optional): The size of the figure in inches. Defaults to (18, 12).

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=figsize)
        markers = ['p']
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']

        pos = nx.get_node_attributes(self.pg.graph, 'pos')
        nx.draw_networkx(self.pg.graph, pos, with_labels=True, node_color='lightblue', node_size=600,font_size=10, font_color='black')
        nx.draw_networkx_edge_labels(self.pg.graph, pos, edge_labels=nx.get_edge_attributes(self.pg.graph, 'weight'), font_size=7)
        
        for i, agent in enumerate(self.agents):
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            if isinstance(self.observations[agent], tuple):  # the agent is transitioning
                pos1, pos2 = [self.pg.getNodePosition(node) for node in self.observations[agent]]
                ratio = self.remaining_steps[agent] / self.pg.graph.edges[self.observations[agent]]['weight']
                pos = (pos1[0]*ratio + pos2[0]*(1-ratio), pos1[1]*ratio + pos2[1]*(1-ratio))
            else:  # the agent is at a node
                pos = self.pg.getNodePosition(self.observations[agent])

            plt.scatter(*pos, color=colors[i % len(colors)], marker=markers[i % len(markers)], zorder=10, alpha=0.3, s=300)
            plt.plot([], [], color=color, marker=marker, linestyle='None', label=agent, alpha=0.5)

        plt.legend()
        plt.text(0,0,f'Current step: {self.step_count}')
        plt.show()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.observations[agent]

    def step(self, action_dict={}):
        """
        Perform a step in the environment based on the given action dictionary.

        Args:
            action_dict (dict): A dictionary containing actions for each agent.

        Returns:
            obs_dict (dict): A dictionary containing the observations for each agent.
            reward_dict (dict): A dictionary containing the rewards for each agent.
            done_dict (dict): A dictionary indicating whether each agent is done.
            info_dict (dict): A dictionary containing additional information for each agent.
        """
        self.step_count += 1
        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        info_dict = {}

        for agent in self.agents:
            # If the agent is at a node, not transitioning
            if self.remaining_steps[agent] == 0 and agent in action_dict:
                action = action_dict[agent]
                if isinstance(self.observations[agent], tuple):
                    current_position = self.observations[agent][0]
                else:
                    current_position = self.observations[agent]

                # If the agent is asked to go where it already is
                if action == current_position:
                    reward_dict[agent] = 0  # you could also give a reward or penalty here if you wanted
                    continue

                # Update the agent's position.
                if action in self.pg.graph.nodes:
                    path = nx.shortest_path(self.pg.graph, source=current_position, target=action, weight='weight')
                    path_len = nx.shortest_path_length(self.pg.graph, source=current_position, target=action, weight='weight')

                    # If there is a direct edge
                    if path_len == 1:
                        self.observations[agent] = action
                        reward_dict[agent] = self.pg.graph.edges[current_position, action]['weight']
                    else:
                        # The agent will start transitioning to the next node on the shortest path
                        self.next_node[agent] = path[1]
                        self.remaining_steps[agent] = self.pg.graph.edges[current_position, self.next_node[agent]]['weight']
                        reward_dict[agent] = 0  # Assuming no reward until it reaches the destination node
                        # Update observation to indicate that agent is transitioning
                        self.observations[agent] = (current_position, self.next_node[agent])
                else:
                    reward_dict[agent] = 0  # the action was invalid
            elif self.remaining_steps[agent] > 0:
                # The agent is transitioning
                self.remaining_steps[agent] -= 10

                # If the agent has reached the next node in its path
                if self.remaining_steps[agent] <= 0:
                    self.observations[agent] = self.next_node[agent]
                    self.next_node[agent] = None

            # Check if the agent is done
            done_dict[agent] = self.dones[agent]

            # Add any additional information for the agent
            info_dict[agent] = {}

            # Update the observation for the agent
            obs_dict[agent] = self.observe(agent)

        return obs_dict, reward_dict, done_dict, {}, info_dict
