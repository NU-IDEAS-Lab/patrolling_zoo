from collections import defaultdict, deque
from itertools import chain
import os
import time
import copy

import imageio
import numpy as np
import torch
import wandb

from onpolicy.utils.shared_buffer import SharedReplayBuffer
from onpolicy.utils.separated_buffer import SeparatedReplayBuffer
from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()

class PatrollingRunner(Runner):
    def __init__(self, config):

        # The default restore functionality is broken. Disable it and do it ourselves.
        model_dir = config['all_args'].model_dir
        config['all_args'].model_dir = None

        super(PatrollingRunner, self).__init__(config)
        self.env_infos = defaultdict(list)

        if self.use_centralized_V:
            # Set up a shared critic. This is slightly hacky!
            # We abuse the existing classses and just use the critic from the first agent.
            self.critic = self.policy[0].critic
            self.critic_optimizer = self.policy[0].critic_optimizer
            for po in self.policy:
                po.critic = self.critic
                po.critic_optimizer = self.critic_optimizer
            for ta in self.trainer:
                if ta._use_popart:
                    ta.value_normalizer = self.critic.v_out
            
            # Set up a shared replay buffer for the critic.
            self.critic_buffer = SharedReplayBuffer(
                self.all_args,
                self.num_agents,
                self.envs.observation_space[0],
                self.envs.share_observation_space[0],
                self.envs.action_space[0]
            )
            
        
        # Set up additional replay buffers for asynchronous actors.
        if self.all_args.skip_steps_async:
            self.buffer = [[] for i in range(self.n_rollout_threads)]
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]

                    # make a copy of all_args but with n_rollout_threads = 1, since we are hacking this to use a separate buffer per rollout thread per agent.
                    args = copy.deepcopy(self.all_args)
                    args.n_rollout_threads = 1
                    bu = SeparatedReplayBuffer(
                        args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id]
                    )
                    self.buffer[i].append(bu)
        
        # Perform restoration.
        config['all_args'].model_dir = model_dir
        self.model_dir = config['all_args'].model_dir
        if self.model_dir is not None:
            self.restore()
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # Create a matrix indicating whether each agent in each environment is ready for a new action.
            self.ready = np.ones((self.n_rollout_threads, self.num_agents), dtype=bool)

            # Set up a temporary buffer to be used for asynchronous skipping.
            tb_actions = -1.0 * np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            tb_action_log_probs = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
            tb_rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            tb_rnn_states_critic = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            tb_value_preds = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            # Set up a reward sum matrix to hold the sum of rewards given between actions (in case of skipping).
            # Currently only used for asynchronous skipping.
            rewardSums = np.zeros((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            # Set the delta steps to 1.
            delta_steps = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)

            # In async mode, reset the buffer for each rollout thread for each agent.
            if self.all_args.skip_steps_async:
                for i in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        self.buffer[i][agent_id].step = 0
                if self.use_centralized_V:
                    self.critic_buffer.step = 0
            
            # No previous data.
            data_critic_prev = None

            step = 0
            while True:
                last_step = self._is_last_step(step)
                if last_step:
                    # If this is the last step, set all agents to ready.
                    self.ready = np.ones((self.n_rollout_threads, self.num_agents), dtype=bool)

                # Sample actions, collect values and probabilities.
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Set up combined action and metadata structure to feed to the environment.
                # We do this to avoid modifying the underlying MAPPO code to add another argument.
                action_metadata = []
                for i in range(self.n_rollout_threads):
                    action_metadata.append({
                        "action": actions_env[i],
                        "metadata": {
                            "last_step": last_step
                        }
                    })
                
                # Take the step and observe.
                combined_obs, rewards, dones, infos = self.envs.step(action_metadata)

                # Process information after taking the step.
                obs, share_obs = self._process_combined_obs(combined_obs)
                for i in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        # Pull agent ready state from the info message.
                        self.ready[i, a] = infos[i]["ready"][a]

                        # Get the number of steps taken by each agent since the agent was last ready.
                        delta_steps = np.array([info["deltaSteps"] for info in infos])

                        # Update the reward sums.
                        rewardSums[i, a] = rewards[i][a]

                        # Update the temporary buffer.
                        if actions[i][a] != None:
                            tb_actions[i, a] = actions[i][a]
                            tb_action_log_probs[i, a] = action_log_probs[i][a]
                            tb_rnn_states[i, a] = rnn_states[i][a]
                            tb_rnn_states_critic[i, a] = rnn_states_critic[i][a]
                            tb_value_preds[i, a] = values[i][a]

                # Combine into single data structure.
                data = obs, share_obs, rewardSums, dones, infos, tb_value_preds, tb_actions, tb_action_log_probs, tb_rnn_states, tb_rnn_states_critic, delta_steps
                
                # Insert data into buffer
                data_critic_prev = self.insert(data, data_critic_prev)

                # Reset the reward sums for any agents that are ready.
                for i in range(self.n_rollout_threads):
                    for a in range(self.num_agents):
                        if self.ready[i, a]:
                            rewardSums[i, a] = 0.0

                # Ensure that all buffers are at step 0.
                if step == 0 and self.all_args.skip_steps_async:
                    for i in range(self.n_rollout_threads):
                        for agent_id in range(self.num_agents):
                            if self.buffer[i][agent_id].step > 1:
                                raise RuntimeError(f"Buffer {i}, agent {agent_id} has step {self.buffer[i][agent_id].step} at the start of an episode. Must be <= 1.")
                if step == 0 and self.use_centralized_V and self.critic_buffer.step > 1:
                    raise RuntimeError(f"Critic buffer has step {self.critic_buffer.step} at the start of an episode. Must be <= 1.")
                
                # Check termination conditions.
                if last_step:
                    break

                # Increase the step count.
                step += 1

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if total_num_steps % self.log_interval == 0:
                end = time.time()
                print("\n Env {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.env_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))
                
                if self.all_args.skip_steps_async:
                    avgEpRewards = np.mean([self.buffer[i][a].rewards for a in range(self.num_agents) for j in range(self.n_rollout_threads)]) * self.episode_length
                else:
                    avgEpRewards = np.mean([self.buffer[a].rewards for a in range(self.num_agents)]) * self.episode_length
                if self.use_wandb:
                    wandb.log({"average_episode_rewards": avgEpRewards}, step=total_num_steps)
                else:
                    self.writter.add_scalars("average_episode_rewards", {"average_episode_rewards": avgEpRewards}, total_num_steps)
                
                avgIdleness = np.mean([self.env_infos["avg_idleness"]])

                print("average episode rewards is {} and idleness is {}".format(avgEpRewards, avgIdleness))
                self.log_train(train_infos, total_num_steps)
                self.log_env(self.env_infos, total_num_steps)
                self.env_infos = defaultdict(list)

            # eval
            if total_num_steps % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        combined_obs = self.envs.reset()

        # Split the combined observations into obs and share_obs, then combine across environments.
        obs, share_obs = self._process_combined_obs(combined_obs)

        if self.use_centralized_V:
            c_share_obs = np.repeat(share_obs, self.num_agents, axis=1)
            c_share_obs = np.reshape(c_share_obs, (self.n_rollout_threads, self.num_agents, -1))
            self.critic_buffer.share_obs[0] = c_share_obs.copy()
            self.critic_buffer.obs[0] = obs.copy()

        # If using asynchronous skipping, warm-start the buffer for each rollout thread for each agent.
        if self.all_args.skip_steps_async:
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    self.buffer[i][agent_id].share_obs[0] = share_obs[i].copy()
                    self.buffer[i][agent_id].obs[0] = np.array(list(obs[i, agent_id])).copy()
        
        # Otherwise, warm-start the buffer for each agent.
        else:
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].share_obs[0] = share_obs.copy()
                self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        actions = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        action_log_probs = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        rnn_states = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        rnn_states_critic = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        actions_env = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]

        # If using centralized critic, get values once for all agents.
        if self.use_centralized_V:
            value, rnn_state_critic = self.trainer[0].policy.get_values_rnn_states(
                np.concatenate(self.critic_buffer.share_obs[self.critic_buffer.step]),
                np.concatenate(self.critic_buffer.rnn_states_critic[self.critic_buffer.step]),
                np.concatenate(self.critic_buffer.masks[self.critic_buffer.step])
            )

        # Get actions from the policy.
        if self.all_args.skip_steps_async:
            # In asynchronous skipping mode, we use individual buffers for each rollout thread and agent.
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    # If the agent is not ready, skip it.
                    if not self.ready[i, agent_id]:
                        if self.use_centralized_V:
                            # We still need to update some data for the centralized critic.
                            values[i][agent_id] = _t2n(value)[0]
                            rnn_states_critic[i][agent_id] = _t2n(rnn_state_critic)[0]
                        continue

                    self.trainer[agent_id].prep_rollout()

                    buf = self.buffer[i][agent_id]
                    v, action, action_log_prob, rnn_state, rsc = self.trainer[agent_id].policy.get_actions(
                        buf.share_obs[buf.step],
                        buf.obs[buf.step],
                        buf.rnn_states[buf.step],
                        buf.rnn_states_critic[buf.step],
                        buf.masks[buf.step]
                    )

                    # Determine whether to use value from centralized critic.
                    if not self.use_centralized_V:
                        value = v
                        rnn_state_critic = rsc

                    values[i][agent_id] = _t2n(value)[0]
                    action = _t2n(action)[0]
                    actions[i][agent_id] = action
                    actions_env[i][agent_id] = action[0]
                    action_log_probs[i][agent_id] = _t2n(action_log_prob)[0]
                    rnn_states[i][agent_id] = _t2n(rnn_state)[0]
                    rnn_states_critic[i][agent_id] = _t2n(rnn_state_critic)[0]

        else:
            # Otherwise, we use a single buffer for each agent across all threads.
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()

                v, action, action_log_prob, rnn_state, rsc = self.trainer[agent_id].policy.get_actions(
                    self.buffer[agent_id].share_obs[step],
                    self.buffer[agent_id].obs[step],
                    self.buffer[agent_id].rnn_states[step],
                    self.buffer[agent_id].rnn_states_critic[step],
                    self.buffer[agent_id].masks[step]
                )

                # Determine whether to use value from centralized critic.
                if not self.use_centralized_V:
                    value = v
                    rnn_state_critic = rsc

                for i in range(self.n_rollout_threads):
                    values[i][agent_id] = _t2n(value)[i]
                    a = _t2n(action)[i]
                    actions[i][agent_id] = a
                    actions_env[i][agent_id] = a[0]
                    action_log_probs[i][agent_id] = _t2n(action_log_prob)[i]
                    rnn_states[i][agent_id] = _t2n(rnn_state)[i]
                    rnn_states_critic[i][agent_id] = _t2n(rnn_state_critic)[i]
        
        actions_env = np.array(actions_env)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data, data_critic):

        # Split data.
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps = data

        # Create a temporary data structure for the critic data.
        # By default, the critic will use the previous data for each agent,
        # unless the agent is ready, in which case it will use the current data.
        if self.use_centralized_V:
            if data_critic == None:
                data_critic = data
            
            c_insert_required = False
            
            # Split data.
            c_obs, c_share_obs, c_rewards, c_dones, c_infos, c_values, c_actions, c_action_log_probs, c_rnn_states, c_rnn_states_critic, c_delta_steps = data_critic

            # Set up the critic shared observation.
            c_share_obs = np.repeat(share_obs[:, np.newaxis, :], self.num_agents, axis=1)
        
        # Add the average idleness time to env infos.
        self.env_infos["avg_idleness"] = [i["avg_idleness"] for i in infos]
        self.env_infos["stddev_idleness"] = [i["stddev_idleness"] for i in infos]
        self.env_infos["worst_idleness"] = [i["worst_idleness"] for i in infos]

        # Add the number of nodes visited to env infos.
        for n in range(len(infos[0]["node_visits"])):
            self.env_infos[f"node_visits/node_{n}"] = [i["node_visits"][n] for i in infos]

        # Reset RNN and mask arguments for done agents/envs.
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if dones[i, agent_id]:
                    rnn_states[i][agent_id] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    rnn_states_critic[i][agent_id] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    masks[i, agent_id] = np.zeros(1, dtype=np.float32)
        
        # If using asynchronous skipping, insert the data into a buffer for each agent for each rollout thread.
        if self.all_args.skip_steps_async:
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    # Only insert if the agent is ready for a new action.
                    if self.ready[i, agent_id]:
                        if self.use_centralized_V:
                            s_obs = share_obs[i]
                            c_insert_required = True
                        else:
                            s_obs = np.array(list(obs[i, agent_id]))

                        # Insert the data into the agent's buffer.
                        self.buffer[i][agent_id].insert(
                            share_obs = s_obs,
                            obs = np.array(list(obs[i, agent_id])),
                            rnn_states = rnn_states[i][agent_id],
                            rnn_states_critic = rnn_states_critic[i][agent_id],
                            actions = actions[i][agent_id],
                            action_log_probs = action_log_probs[i][agent_id],
                            value_preds = values[i][agent_id],
                            rewards = rewards[i, agent_id],
                            masks = masks[i, agent_id],
                            deltaSteps = delta_steps[i, agent_id],
                            criticStep = np.array(self.critic_buffer.step) if self.use_centralized_V else np.array(self.buffer[i][agent_id].step),
                            no_reset = True
                        )
                    
                    # If we are using a shared critic, update the critic data.
                    if self.use_centralized_V:
                        # Update these regardless of whether the agent was skipped.
                        c_obs[i, agent_id] = obs[i, agent_id]
                        c_rewards[i, agent_id] = rewards[i, agent_id]
                        c_values[i][agent_id] = values[i][agent_id]
                        c_rnn_states_critic[i][agent_id] = rnn_states_critic[i][agent_id]
                        # c_delta_steps[i, agent_id] = 1

                        # Only update these if the agent was not skipped.
                        if actions[i][agent_id] != None:
                            c_rnn_states[i][agent_id] = rnn_states[i][agent_id]
                            c_actions[i][agent_id] = actions[i][agent_id]
                            c_action_log_probs[i][agent_id] = action_log_probs[i][agent_id]
        
        # Otherwise, insert the data into a buffer for each agent.
        else:
            c_insert_required = True

            # Convert to numpy arrays. Until this point, the data is in Python arrays to allow for ragged arrays (useful for async skipping).
            rnn_states = np.array(rnn_states)
            rnn_states_critic = np.array(rnn_states_critic)
            actions = np.array(actions)
            action_log_probs = np.array(action_log_probs)
            values = np.array(values)
            rewards = np.array(rewards)
            masks = np.array(masks)
            delta_steps = np.array(delta_steps)
            if self.use_centralized_V:
                c_obs = np.array(c_obs)
                c_rnn_states = np.array(c_rnn_states)
                c_rnn_states_critic = np.array(c_rnn_states_critic)
                c_actions = np.array(c_actions)
                c_action_log_probs = np.array(c_action_log_probs)
                c_values = np.array(c_values)
                c_rewards = np.array(c_rewards)
                c_delta_steps = np.array(c_delta_steps)

            # Insert for every agent.
            for agent_id in range(self.num_agents):
                if self.use_centralized_V:
                    s_obs = share_obs
                else:
                    s_obs = np.array(list(obs[:, agent_id]))
                
                self.buffer[agent_id].insert(
                    share_obs = s_obs,
                    obs = np.array(list(obs[:, agent_id])),
                    rnn_states = rnn_states[:, agent_id],
                    rnn_states_critic = rnn_states_critic[:, agent_id],
                    actions = actions[:, agent_id],
                    action_log_probs = action_log_probs[:, agent_id],
                    value_preds = values[:, agent_id],
                    rewards = rewards[:, agent_id],
                    masks = masks[:, agent_id],
                    deltaSteps = delta_steps[:, agent_id]
                )
                
                # If we are using a shared critic, update the critic data.
                if self.use_centralized_V:
                    c_obs[:, agent_id] = obs[:, agent_id]
                    c_rnn_states[:, agent_id] = rnn_states[:, agent_id]
                    c_rnn_states_critic[:, agent_id] = rnn_states_critic[:, agent_id]
                    c_actions[:, agent_id] = actions[:, agent_id]
                    c_action_log_probs[:, agent_id] = action_log_probs[:, agent_id]
                    c_values[:, agent_id] = values[:, agent_id]
                    c_rewards[:, agent_id] = rewards[:, agent_id]
                    c_delta_steps[:, agent_id] = delta_steps[:, agent_id]

        # If we are using a shared critic, insert the critic data.
        if self.use_centralized_V:
            if c_insert_required:
                # Reset RNN and mask arguments for done agents/envs.
                c_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                for i in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        if dones[i, agent_id]:
                            c_rnn_states[i][agent_id] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                            c_rnn_states_critic[i][agent_id] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                            c_masks[i, agent_id] = np.zeros((1, 1), dtype=np.float32)

                self.critic_buffer.insert(
                    share_obs=np.array(c_share_obs),
                    obs=np.array(c_obs),
                    rnn_states=np.array(c_rnn_states),
                    rnn_states_critic=np.array(c_rnn_states_critic),
                    actions=np.array(c_actions),
                    action_log_probs=np.array(c_action_log_probs),
                    value_preds=np.array(c_values),
                    rewards=np.array(c_rewards),
                    masks=c_masks,
                    deltaSteps=np.array(c_delta_steps),
                    no_reset=self.all_args.skip_steps_async
                )

                c_delta_steps = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            
            # If critic update not required, increment the step counter.
            else:
                c_delta_steps += 1

            # Update the previous data.
            data_critic = c_obs, c_share_obs, c_rewards, c_dones, c_infos, c_values, c_actions, c_action_log_probs, c_rnn_states, c_rnn_states_critic, c_delta_steps
        
        return data_critic

    def log_train(self, train_infos, total_num_steps): 
        # The train_infos is a list (size self.n_rollout_threads) of lists (size self.num_agents) of dicts.
        # We want to flatten this to a single list of dicts by averaging across rollout threads.
        train_infos_actors = [dict(chain.from_iterable(d.items() for d in agent_infos)) for agent_infos in zip(*train_infos["actors"])]

        # Log actors.
        for agent_id in range(self.num_agents):
            for k, v in train_infos_actors[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
        
        # Log critic.
        if self.use_centralized_V:
            train_infos_critic = dict(chain.from_iterable(d.items() for d in train_infos["critic"]))
            for k, v in train_infos_critic.items():
                if self.use_wandb:
                    wandb.log({k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: v}, total_num_steps)


    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if type(v) == wandb.viz.CustomChart and self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            elif len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self, ipython_clear_output=True):        

        if ipython_clear_output:
            from IPython.display import clear_output

        # reset envs and init rnn and mask
        render_env = self.envs

        for i_episode in range(self.all_args.render_episodes):
            combined_obs = render_env.reset()
            render_actions = np.zeros((self.n_render_rollout_threads, self.num_agents), dtype=np.float32)
            render_rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            render_masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            # Split the combined observations into obs and share_obs, then combine across environments.
            obs = []
            share_obs = []
            for o in combined_obs:
                obs.append(o["obs"])
                share_obs.append(o["share_obs"][0])
            obs = np.array(obs)
            share_obs = np.array(share_obs)

            # Record the readiness of each agent for a new action. All agents ready by default.
            ready = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=bool)


            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                # We only use a single rollout thread for rendering.
                for i in range(self.n_render_rollout_threads):
                    for agent_id in range(self.num_agents):
                        if not ready[i, agent_id]:
                            continue

                        if agent_id == 0:
                            print('hello')

                        self.trainer[agent_id].prep_rollout()
                        render_action, render_rnn_state = self.trainer[agent_id].policy.act(obs[:, agent_id],
                                                                            render_rnn_states[:, agent_id],
                                                                            render_masks[:, agent_id],
                                                                            deterministic=True)

                        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                        render_actions[i, agent_id] = np.array(np.split(_t2n(render_action), self.n_render_rollout_threads))
                        render_rnn_states[i, agent_id] = np.array(np.split(_t2n(render_rnn_state), self.n_render_rollout_threads))

                render_actions_env = [render_actions[idx, :] for idx in range(self.n_render_rollout_threads)]

                print(render_actions_env[0], ready[0])

                # step
                combined_obs, render_rewards, render_dones, render_infos = render_env.step(render_actions_env)

                # Split the combined observations into obs and share_obs, then combine across environments.
                obs = []
                share_obs = []
                for o in combined_obs:
                    obs.append(o["obs"])
                    share_obs.append(o["share_obs"][0])
                obs = np.array(obs)
                share_obs = np.array(share_obs)

                # Pull agent ready state from the info message.
                for i in range(self.n_render_rollout_threads):
                    for a in range(self.num_agents):
                        ready[i, a] = render_infos[i]["ready"][a]

                # Display with ipython
                if not np.any(render_dones):
                    if ipython_clear_output:
                        clear_output(wait = True)
                    render_env.envs[0].env.render()

                # append frame
                if self.all_args.save_gifs:        
                    image = render_infos[0]["frame"]
                    frames.append(image)

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )

    @torch.no_grad()
    def compute(self):

        # Compute returns for the centralized critic.
        if self.use_centralized_V:
            self.trainer[0].prep_rollout()

            buf = self.critic_buffer

            # Check that the total step count is correct.
            if self.critic_buffer.step != self.all_args.episode_length:
                raise RuntimeError(f"Total step count is incorrect for critic buffer! Expected {self.all_args.episode_length}, got {self.critic_buffer.step}")
            # stepSum = np.sum(buf.deltaSteps[:buf.step], axis=0)
            # if not self.all_args.skip_steps_sync and np.any(stepSum != self.all_args.episode_length):
            #     raise RuntimeError(f"Total step count is incorrect for critic buffer! Expected {self.all_args.episode_length}, got {stepSum}")

            next_value = self.trainer[0].policy.get_values(np.concatenate(buf.share_obs[buf.step - 1]), 
                                                            np.concatenate(buf.rnn_states_critic[buf.step - 1]),
                                                            np.concatenate(buf.masks[buf.step - 1]))
            next_value = np.array(np.split(_t2n(next_value), self.n_rollout_threads))
            self.critic_buffer.compute_returns(
                next_value,
                self.trainer[0].value_normalizer,
                last_step=buf.step
            )

            # Copy value predictions and returns to each agent's buffer.
            if self.all_args.skip_steps_async:
                for i in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        for j in range(self.buffer[i][agent_id].step):
                            s = self.buffer[i][agent_id].criticStep[j]
                            self.buffer[i][agent_id].returns[j] = self.critic_buffer.returns[s, i, agent_id]

                        # Copy values for last step to capture the final value/return.
                        # self.buffer[i][agent_id].returns[j + 1] = self.critic_buffer.returns[self.critic_buffer.step, i, agent_id]
                        # self.buffer[i][agent_id].value_preds[j + 1] = self.critic_buffer.value_preds[self.critic_buffer.step, i, agent_id]
            else:
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].returns = self.critic_buffer.returns[:, :, agent_id]

        # Compute returns for the decentralized critics.
        else:
            # If using asynchronous skipping, we compute returns for each agent's policy using buffers from each rollout thread.
            if self.all_args.skip_steps_async:
                for i in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        self.trainer[agent_id].prep_rollout()
                        buf = self.buffer[i][agent_id]

                        # # Check that the total step count is correct.
                        # stepSum = np.sum(buf.deltaSteps[:buf.step])
                        # if stepSum != self.all_args.episode_length:
                        #     raise RuntimeError(f"Total step count is incorrect for buffer {i}-{agent_id}! Expected {self.all_args.episode_length}, got {stepSum}")

                        next_value = self.trainer[agent_id].policy.get_values(
                            buf.share_obs[buf.step - 1], 
                            buf.rnn_states_critic[buf.step - 1],
                            buf.masks[buf.step - 1]
                        )
                        next_value = _t2n(next_value)
                        buf.compute_returns(
                            next_value,
                            self.trainer[agent_id].value_normalizer,
                            last_step=buf.step
                        )
            
            # Otherwise, we compute returns for each agent's policy using a single buffer.
            else:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    buf = self.buffer[agent_id]

                    # Check that the total step count is correct.
                    stepSum = np.sum(buf.deltaSteps[:buf.step], axis=0)
                    if not self.all_args.skip_steps_sync and np.any(stepSum != self.all_args.episode_length):
                        raise RuntimeError(f"Total step count is incorrect for buffer {agent_id}! Expected {self.all_args.episode_length}, got {stepSum}")

                    next_value = self.trainer[agent_id].policy.get_values(
                        buf.share_obs[buf.step - 1],
                        buf.rnn_states_critic[buf.step - 1],
                        buf.masks[buf.step - 1]
                    )
                    next_value = _t2n(next_value)
                    buf.compute_returns(
                        next_value,
                        self.trainer[agent_id].value_normalizer,
                        last_step=buf.step
                    )

    def train(self):
        train_infos = {
            "actors": [[] for i in range(self.n_rollout_threads)],
            "critic": []
        }

        # Update the centralized critic.
        if self.use_centralized_V:
            self.trainer[0].prep_training()
            train_info = self.trainer[0].train(
                self.critic_buffer,
                update_actor=False,
                update_critic=True,
                last_step=self.critic_buffer.step
            )
            train_infos["critic"].append(train_info)
            self.critic_buffer.after_update(
                last_step = self.critic_buffer.step
            )

        # Print the entire actor buffer.
        # buf = self.critic_buffer
        # # buf = self.buffer[0][0]
        # for at in ["obs", "share_obs", "actions", "value_preds", "returns", "masks", "rnn_states", "rnn_states_critic", "deltaSteps"]:
        #     print(f"======= ATTRIBUTE: {at} =======")
        #     for i in buf.__dict__[at]:
        #         print(i)
        #     print()
        # raise Exception("Done printing buffer.")

        # Update the actors (and critics in case of per-agent critics).
        if self.all_args.skip_steps_async:
            # In case of asynchronous skipping, we update each agent's policy using buffers from each rollout thread.
            for i in range(self.n_rollout_threads):
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_training()
                    train_info = self.trainer[agent_id].train(
                        self.buffer[i][agent_id],
                        update_actor=True,
                        update_critic=(not self.use_centralized_V),
                        last_step=self.buffer[i][agent_id].step
                    )
                    train_infos["actors"][i].append(train_info)       
                    self.buffer[i][agent_id].after_update(
                        last_step = self.buffer[i][agent_id].step
                    )
        
        else:
            # Otherwise, we update each agent's policy using a single buffer.
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_training()
                train_info = self.trainer[agent_id].train(
                    self.buffer[agent_id],
                    update_actor=True,
                    update_critic=(not self.use_centralized_V)
                )
                for i in range(self.n_rollout_threads):
                    train_infos["actors"][i].append(train_info)       
                self.buffer[agent_id].after_update()

        return train_infos
    
    def _is_last_step(self, step):
        ''' Determine whether this step is the last step of the episode. '''

        if self.all_args.skip_steps_async:
            if self.use_centralized_V:
                bufferStep = self.critic_buffer.step
            else:
                maxBufferStep = 0
                for i in range(self.n_rollout_threads):
                    for agent_id in range(self.num_agents):
                        maxBufferStep = max(maxBufferStep, self.buffer[i][agent_id].step)
                bufferStep = maxBufferStep
            last_step = bufferStep >= self.episode_length - 1
        else:
            last_step = step >= self.episode_length - 1
        return last_step

    def _process_combined_obs(self, combined_obs):
        ''' Process the combined observations into obs and share_obs. '''
        obs = []
        share_obs = []
        for o in combined_obs:
            obs.append(o["obs"])
            share_obs.append(o["share_obs"])
        return np.array(obs), np.array(share_obs)