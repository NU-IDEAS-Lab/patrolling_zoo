from collections import defaultdict, deque
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
import wandb

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()

class PatrollingRunner(Runner):
    def __init__(self, config):

        # The default restore functionality is broken. Disable it and do it ourselves.
        model_dir = config['all_args'].model_dir
        config['all_args'].model_dir = None

        super(PatrollingRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
       
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
                self.trainer.policy.lr_decay(episode, episodes)

            # Set the delta steps to 1.
            delta_steps = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.int32)
            for step in range(self.episode_length):
                # Sample actions
                # Sample actions, collect values and probabilities.
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                combined_obs, rewards, dones, infos = self.envs.step(actions_env)

                # Split the combined observations into obs and share_obs, then combine across environments.
                obs, share_obs = self._process_combined_obs(combined_obs)

                # Get the number of steps taken by each agent since the agent was last ready.
                delta_steps = np.array([info["deltaSteps"] for info in infos])

                data = obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps
                
                # insert data into buffer
                self.insert(data)

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
                
                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {} and idleness is {}".format(train_infos["average_episode_rewards"], np.mean(self.env_infos["avg_idleness"])))
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

        # insert obs to buffer
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()

        values = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        actions = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        action_log_probs = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        rnn_states = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        rnn_states_critic = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]
        actions_env = [[None for a in range(self.num_agents)] for idx in range(self.n_rollout_threads)]

        for agent_id in range(self.num_agents):
            self.trainer.prep_rollout()

            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer.policy.get_actions(
                self.buffer.share_obs[step],
                self.buffer.obs[step, :, agent_id],
                self.buffer.rnn_states[step],
                self.buffer.rnn_states_critic[step],
                self.buffer.masks[step]
            )

            for i in range(self.n_rollout_threads):
                values[i][agent_id] = _t2n(value)[i]
                a = _t2n(action)[i]
                actions[i][agent_id] = a
                actions_env[i][agent_id] = a[0]
                action_log_probs[i][agent_id] = _t2n(action_log_prob)[i]
                rnn_states[i][agent_id] = _t2n(rnn_state)[i]
                rnn_states_critic[i][agent_id] = _t2n(rnn_state_critic)[i]
        
        values = np.array(values)
        actions = np.array(actions)
        action_log_probs = np.array(action_log_probs)
        rnn_states = np.array(rnn_states)
        rnn_states_critic = np.array(rnn_states_critic)
        actions_env = np.array(actions_env)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, delta_steps = data
        
        # update env_infos if done
        dones_env = np.all(dones, axis=-1)

        # Add the average idleness time to env infos.
        self.env_infos["avg_idleness"] = [i["avg_idleness"] for i in infos]
        self.env_infos["stddev_idleness"] = [i["stddev_idleness"] for i in infos]
        self.env_infos["worst_idleness"] = [i["worst_idleness"] for i in infos]
        self.env_infos["agent_count"] = [i["agent_count"] for i in infos]

        # Add the number of nodes visited to env infos.
        for n in range(len(infos[0]["node_visits"])):
            self.env_infos[f"node_visits/node_{n}"] = [i["node_visits"][n] for i in infos]

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        for i in range(self.n_rollout_threads):
            for agent_id in range(self.num_agents):
                if dones[i, agent_id]:
                    rnn_states[i][agent_id] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    rnn_states_critic[i][agent_id] = np.zeros((self.recurrent_N, self.hidden_size), dtype=np.float32)
                    masks[i, agent_id] = np.zeros(1, dtype=np.float32)

        self.buffer.insert(
            share_obs=share_obs,
            obs=obs,
            rnn_states=rnn_states,
            rnn_states_critic=rnn_states_critic,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=values,
            rewards=rewards,
            masks=masks,
            deltaSteps=delta_steps
        )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if type(v) == wandb.viz.CustomChart and self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            elif len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v, axis=0)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)    

    @torch.no_grad()
    def eval(self, total_num_steps):
        # reset envs and init rnn and mask
        eval_obs = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # init eval goals
        num_done = 0
        eval_goals = np.zeros(self.all_args.eval_episodes)
        eval_win_rates = np.zeros(self.all_args.eval_episodes)
        eval_steps = np.zeros(self.all_args.eval_episodes)
        step = 0
        quo = self.all_args.eval_episodes // self.n_eval_rollout_threads
        rem = self.all_args.eval_episodes % self.n_eval_rollout_threads
        done_episodes_per_thread = np.zeros(self.n_eval_rollout_threads, dtype=int)
        eval_episodes_per_thread = done_episodes_per_thread + quo
        eval_episodes_per_thread[:rem] += 1
        unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

        # loop until enough episodes
        while num_done < self.all_args.eval_episodes and step < self.episode_length:
            # get actions
            self.trainer.prep_rollout()

            # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=self.all_args.eval_deterministic
            )
            
            # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
            eval_actions = np.array(np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = [eval_actions[idx, :, 0] for idx in range(self.n_eval_rollout_threads)]

            # step
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

            # update goals if done
            eval_dones_env = np.all(eval_dones, axis=-1)
            eval_dones_unfinished_env = eval_dones_env[unfinished_thread]
            if np.any(eval_dones_unfinished_env):
                for idx_env in range(self.n_eval_rollout_threads):
                    if unfinished_thread[idx_env] and eval_dones_env[idx_env]:
                        eval_goals[num_done] = eval_infos[idx_env]["score_reward"]
                        eval_win_rates[num_done] = 1 if eval_infos[idx_env]["score_reward"] > 0 else 0
                        eval_steps[num_done] = eval_infos[idx_env]["max_steps"] - eval_infos[idx_env]["steps_left"]
                        # print("episode {:>2d} done by env {:>2d}: {}".format(num_done, idx_env, eval_infos[idx_env]["score_reward"]))
                        num_done += 1
                        done_episodes_per_thread[idx_env] += 1
            unfinished_thread = (done_episodes_per_thread != eval_episodes_per_thread)

            # reset rnn and masks for done envs
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
            step += 1

        # get expected goal
        eval_goal = np.mean(eval_goals)
        eval_win_rate = np.mean(eval_win_rates)
        eval_step = np.mean(eval_steps)
    
        # log and print
        print("eval expected goal is {}.".format(eval_goal))
        if self.use_wandb:
            wandb.log({"eval_goal": eval_goal}, step=total_num_steps)
            wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
            wandb.log({"eval_step": eval_step}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_goal", {"expected_goal": eval_goal}, total_num_steps)
            self.writter.add_scalars("eval_win_rate", {"eval_win_rate": eval_win_rate}, total_num_steps)
            self.writter.add_scalars("eval_step", {"expected_step": eval_step}, total_num_steps)

    @torch.no_grad()
    def render(self, ipython_clear_output=True):        

        if ipython_clear_output:
            from IPython.display import clear_output

        # reset envs and init rnn and mask
        render_env = self.envs

        # init goal
        render_goals = np.zeros(self.all_args.render_episodes)
        for i_episode in range(self.all_args.render_episodes):
            combined_obs = render_env.reset()
            rnn_states = np.zeros((self.n_render_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_render_rollout_threads, self.num_agents, 1), dtype=np.float32)

            # Split the combined observations into obs and share_obs, then combine across environments.
            obs, share_obs = self._process_combined_obs(combined_obs)

            if self.all_args.save_gifs:        
                frames = []
                image = self.envs.envs[0].env.unwrapped.observation()[0]["frame"]
                frames.append(image)

            dones = False
            while not np.any(dones):
                self.trainer.prep_rollout()
                actions, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True
                )

                # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
                actions = np.array(np.split(_t2n(actions), self.n_render_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_render_rollout_threads))

                actions_env = [actions[idx, :, 0] for idx in range(self.n_render_rollout_threads)]

                # step
                combined_obs, render_rewards, dones, infos = render_env.step(actions_env)

                # Split the combined observations into obs and share_obs, then combine across environments.
                obs, share_obs = self._process_combined_obs(combined_obs)

                if not np.any(dones):
                    if ipython_clear_output:
                        clear_output(wait = True)
                    render_env.envs[0].env.render()

                # append frame
                if self.all_args.save_gifs:        
                    image = infos[0]["frame"]
                    frames.append(image)

            # save gif
            if self.all_args.save_gifs:
                imageio.mimsave(
                    uri="{}/episode{}.gif".format(str(self.gif_dir), i_episode),
                    ims=frames,
                    format="GIF",
                    duration=self.all_args.ifi,
                )
    
    def _process_combined_obs(self, combined_obs):
        ''' Process the combined observations into obs and share_obs. '''
        obs = []
        share_obs = []
        for o in combined_obs:
            obs.append(o["obs"])
            share_obs.append(o["share_obs"])

        return np.array(obs), np.array(share_obs)
