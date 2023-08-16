import random

from patrolling_zoo.env.patrolling_zoo import parallel_env
from patrolling_zoo.env.patrol_graph import PatrolGraph
from gymnasium.spaces.utils import flatten, flatten_space
import numpy as np


class PatrollingEnv(object):
    '''Wrapper to make the Patrolling Zoo environment compatible'''

    def __init__(self, args):
        self.num_agents = args.num_agents
        
        # # make env
        # if not (args.use_render and args.save_videos):
        #     self.env = foot.create_environment(
        #         env_name=args.scenario_name,
        #         stacked=args.use_stacked_frames,
        #         representation=args.representation,
        #         rewards=args.rewards,
        #         number_of_left_players_agent_controls=args.num_agents,
        #         number_of_right_players_agent_controls=0,
        #         channel_dimensions=(args.smm_width, args.smm_height),
        #         render=(args.use_render and args.save_gifs)
        #     )
        # else:
        #     # render env and save videos
        #     self.env = football_env.create_environment(
        #         env_name=args.scenario_name,
        #         stacked=args.use_stacked_frames,
        #         representation=args.representation,
        #         rewards=args.rewards,
        #         number_of_left_players_agent_controls=args.num_agents,
        #         number_of_right_players_agent_controls=0,
        #         channel_dimensions=(args.smm_width, args.smm_height),
        #         # video related params
        #         write_full_episode_dumps=True,
        #         render=True,
        #         write_video=True,
        #         dump_frequency=1,
        #         logdir=args.video_dir
        #     )

        pg = PatrolGraph(args.graph_file)

        self.env = parallel_env(
            patrol_graph = pg,
            num_agents = args.num_agents,
            # comms_model = args.comms_model,
            # require_explicit_visit = args.require_explicit_visit,
            # speed = args.speed,
            speed = 10,
            # alpha = args.alpha,
            # observation_radius = args.observation_radius,
            # observe_method = args.observe_method,
            max_cycles = args.num_env_steps,
            observe_method = "ajg_new"
        )
            
        self.max_steps = self.env.max_cycles
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # Set up spaces.
        self.action_space = [flatten_space(self.env.action_spaces[a]) for a in self.env.possible_agents]
        self.observation_space = [flatten_space(self.env.observation_spaces[a]) for a in self.env.possible_agents]
        self.share_observation_space = [flatten_space(self.env.state_space) for a in self.env.possible_agents]

    def reset(self):
        obs, _ = self.env.reset()
        obs = self._obs_wrapper(obs)
        return obs

    def step(self, action):

        action = {self.env.possible_agents[i]: action[i] for i in range(self.num_agents)}
        obs, reward, done, trunc, info = self.env.step(action)
        obs = self._obs_wrapper(obs)
        reward = [reward[a] for a in self.env.possible_agents]
        if self.share_reward:
            global_reward = np.sum(reward)
            reward = [[global_reward]] * self.num_agents

        done = [done[a] for a in self.env.possible_agents]
        info = self._info_wrapper(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.close()

    def _obs_wrapper(self, obs):

        # Flatten the PZ observation.
        obs = flatten(self.env.observation_spaces, obs)
        obs = np.reshape(obs, (self.num_agents, -1))

        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

    def _info_wrapper(self, info):
        # state = self.env.state()
        # info.update(state[0])
        # info["max_steps"] = self.max_steps
        # info["active"] = np.array([state[i]["active"] for i in range(self.num_agents)])
        # info["designated"] = np.array([state[i]["designated"] for i in range(self.num_agents)])
        # info["sticky_actions"] = np.stack([state[i]["sticky_actions"] for i in range(self.num_agents)])
        return info
