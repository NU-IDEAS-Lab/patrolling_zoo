import random

from patrolling_zoo.env.patrolling_zoo import parallel_env
from patrolling_zoo.env.patrol_graph import PatrolGraph
from gymnasium.spaces.utils import flatten, flatten_space
import numpy as np


class PatrollingEnv(object):
    '''Wrapper to make the Patrolling Zoo environment compatible'''

    def __init__(self, args):
        self.args = args
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
            speed = args.agent_speed,
            alpha = args.alpha,
            beta = args.beta,
            observation_radius = args.observation_radius,
            observe_method = args.observe_method,
            observe_method_global = args.observe_method_global,
            reward_method_terminal = args.reward_method_terminal,
            max_cycles = -1 if self.args.skip_steps_sync or self.args.skip_steps_async else args.episode_length
        )
            
        self.remove_redundancy = args.remove_redundancy
        self.zero_feature = args.zero_feature
        self.share_reward = args.share_reward
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # Set up spaces.
        self.action_space = [self.env.action_spaces[a] for a in self.env.possible_agents]
        self.observation_space = [flatten_space(self.env.observation_spaces[a]) for a in self.env.possible_agents]
        self.share_observation_space = [flatten_space(self.env.state_space) for a in self.env.possible_agents]


    def reset(self):
        self.ppoSteps = 0
        self.prevAction = {a: None for a in self.env.possible_agents}
        self.deltaSteps = {a: 0 for a in self.env.possible_agents}
        obs, _ = self.env.reset()
        obs = self._obs_wrapper(obs)

        combined_obs = {
            "obs": obs,
            "share_obs": self._share_obs_wrapper(self.env.state_all())
        }

        return combined_obs

    def step(self, action_metadata):

        ready = False
        done = []
        if type(action_metadata) == dict:
            action = action_metadata["action"]
            last_step = False
            if "metadata" in action_metadata:
                metadata = action_metadata["metadata"]
                last_step = metadata["last_step"]
        else:
            action = action_metadata
            last_step = False


        # Start with the previous action.
        actionPz = self.prevAction

        # For any agents which are ready, use the new action.
        for i in range(self.num_agents):
            if action[i] != None:
                actionPz[self.env.possible_agents[i]] = action[i]

                # Reset step count.
                self.deltaSteps[self.env.possible_agents[i]] = 0
            elif not self.args.skip_steps_async:
                raise ValueError(f"Action cannot be None when skip_steps_async is False. Agent: {i}")

        rewards = np.zeros((self.num_agents, 1), dtype=np.float32)

        while not ready and not any(done):
            # We want to determine if this is the last step when using syncronized step skipping.
            lastStep = last_step or (self.args.skip_steps_sync and self.ppoSteps >= self.args.episode_length - 1)
            
            # Take a step.
            obs, reward, done, trunc, info = self.env.step(actionPz, lastStep=lastStep)

            # Convert the done dict to a list.
            done = [done[a] for a in self.env.possible_agents]
            # Convert the trunc dict to a list.
            trunc = [trunc[a] for a in self.env.possible_agents]

            # Consider the episode done if any agent is done OR truncated.
            done = [d or t for d, t in zip(done, trunc)]

            combined_obs = {
                "obs": self._obs_wrapper(obs),
                "share_obs": self._share_obs_wrapper(self.env.state_all())
            }

            # Increase reward.
            rewards += np.array([reward[a] for a in self.env.possible_agents]).reshape(-1, 1)

            info = self._info_wrapper(info)

            # Increase the step count.
            for a in self.env.possible_agents:
                self.deltaSteps[a] += 1

            # Only run once if skip_steps_sync is false.
            if not self.args.skip_steps_sync:
                break

            # Check if any agents are ready
            ready = any([info[a]["ready"] for a in self.env.agents])

        # If we are sharing the reward, then we need to sum the rewards.
        if self.share_reward:
            global_reward = np.sum(rewards)
            rewards = global_reward * np.ones((self.num_agents, 1), dtype=np.float32)
        
        # Convert back from numpy to list.
        rewards = [rewards[i] for i in range(self.num_agents)]

        info["deltaSteps"] = [[self.deltaSteps[a]] for a in self.env.possible_agents]
        info["ready"] = [info[a]["ready"] for a in self.env.possible_agents]

        self.ppoSteps += 1

        # Update the previous action.
        self.prevAction = actionPz

        return combined_obs, rewards, done, info

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
        
        return obs
    
    def _share_obs_wrapper(self, obs):
        res = []
        for a in self.env.possible_agents:
            res.append(flatten(self.env.state_space, obs[a]))
        res = np.reshape(res, (self.num_agents, -1))
        return res

        #This older code below is for use with the state() method. Above code is for the state_all() method.
        # Flatten the PZ observation.
        # obs = flatten(self.env.state_space, obs)
        # obs = np.repeat(obs[np.newaxis, :], self.num_agents, axis=0)
        # return obs

    def _info_wrapper(self, info):
        return info
