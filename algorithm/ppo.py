from gymnasium.spaces.utils import flatten, flatten_space
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
from IPython.display import clear_output


from .algorithm import BaseAlgorithm

class PPO(BaseAlgorithm):
    ''' This algorithm implements PPO for the patrolling problem.
        Adapted from an example by Jet (https://github.com/jjshoots). '''

    def __init__(self, env, device,
            ent_coef = 0.1,
            vf_coef = 0.1,
            clip_coef = 0.1,
            gamma = 0.99,
            gae_lambda = 0.95,
            lr = 0.0001,
            batch_size = 1,
            stack_size = 4,
            frame_size = (64, 64),
            max_cycles = 500,
            total_episodes = 40
        ):
        super().__init__(env, device)

        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_coef = clip_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.lr = lr
        self.batch_size = batch_size
        self.stack_size = stack_size
        self.frame_size = frame_size
        self.max_cycles = max_cycles
        self.total_episodes = total_episodes

        self.num_agents = len(env.possible_agents)
        self.num_actions = env.action_space(env.possible_agents[0]).n
        self.observation_size = flatten_space(env.observation_space(env.possible_agents[0])).shape[0]

        """ LEARNER SETUP """
        self.learner = PPONetwork(num_actions=self.num_actions, num_agents=self.num_agents, observation_size=self.observation_size, device=self.device).to(self.device)
        self.optimizer = optim.Adam(self.learner.parameters(), lr=self.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_episodes, eta_min=0, last_epoch=-1)
    
    def train(self, seed=None):
        ''' Trains the policy. '''

        # Stats storage
        stats = {
            "episodic_return": [],
            "episodic_length": [],
            "value_loss": [],
            "policy_loss": [],
            "entropy_loss": [],
            "total_loss": [],
            "approx_kl": [],
            "clip_frac": [],
        }

        """ ALGO LOGIC: EPISODE STORAGE"""
        end_step = 0
        total_episodic_return = 0
        rb_obs = torch.zeros((self.max_cycles, self.num_agents, self.observation_size)).to(self.device)
        rb_actions = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        rb_logprobs = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        rb_rewards = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        rb_terms = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)
        rb_values = torch.zeros((self.max_cycles, self.num_agents)).to(self.device)

        # train for n number of episodes
        for episode in range(self.total_episodes):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                next_obs, info = self.env.reset(seed=seed)

                # reset the episodic return
                total_episodic_return = 0

                # each episode has num_steps
                for step in range(0, self.max_cycles):
                    # rollover the observation
                    
                    obs = self.learner.batchify_obs(self.env.observation_space(self.env.possible_agents[0]), next_obs, self.device)

                    actions, logprobs, _, values = self.learner.get_action_and_value(obs)

                    # execute the environment and log data
                    next_obs, rewards, terms, truncs, infos = self.env.step(
                        self.learner.unbatchify(actions, self.env)
                    )

                    # add to episode storage
                    rb_obs[step] = torch.reshape(obs, (self.num_agents, self.observation_size))
                    rb_rewards[step] = self.learner.batchify(rewards, self.device)
                    rb_terms[step] = self.learner.batchify(terms, self.device)
                    rb_actions[step] = actions
                    rb_logprobs[step] = logprobs
                    rb_values[step] = values.flatten()

                    # compute episodic return
                    total_episodic_return += rb_rewards[step].cpu().numpy()

                    # if we reach termination or truncation, end
                    if any([terms[a] for a in terms]) or any([truncs[a] for a in truncs]):
                        end_step = step
                        break

            # bootstrap value if not done
            with torch.no_grad():
                rb_advantages = torch.zeros_like(rb_rewards).to(self.device)
                for t in reversed(range(end_step)):
                    delta = (
                        rb_rewards[t]
                        + self.gamma * rb_values[t + 1] * rb_terms[t + 1]
                        - rb_values[t]
                    )
                    rb_advantages[t] = delta + self.gamma * self.gae_lambda * rb_advantages[t + 1]
                rb_returns = rb_advantages + rb_values

            # convert our episodes to batch of individual transitions
            b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
            b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
            b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
            b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
            b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
            b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

            # Optimizing the policy and value network
            b_index = np.arange(len(b_obs))
            clip_fracs = []
            for repeat in range(3):
                # shuffle the indices we use to access the data
                np.random.shuffle(b_index)
                for start in range(0, len(b_obs), self.batch_size):
                    # select the indices we want to train on
                    end = start + self.batch_size
                    batch_index = b_index[start:end]

                    _, newlogprob, entropy, value = self.learner.get_action_and_value(
                        b_obs[batch_index], b_actions.long()[batch_index]
                    )
                    logratio = newlogprob - b_logprobs[batch_index]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_fracs += [
                            ((ratio - 1.0).abs() > self.clip_coef).float().mean().item()
                        ]

                    # normalize advantaegs
                    advantages = b_advantages[batch_index]
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                    # Policy loss
                    pg_loss1 = -advantages * ratio
                    pg_loss2 = -advantages * torch.clamp(
                        ratio, 1 - self.clip_coef, 1 + self.clip_coef
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value = value.flatten()
                    v_loss_unclipped = (value - b_returns[batch_index]) ** 2
                    v_clipped = b_values[batch_index] + torch.clamp(
                        value - b_values[batch_index],
                        -self.clip_coef,
                        self.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[batch_index]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    # Store statistics.
                    stats["value_loss"] += [v_loss.item()]
                    stats["policy_loss"] += [pg_loss.item()]
                    stats["entropy_loss"] += [entropy_loss.item()]
                    stats["total_loss"] += [loss.item()]
                    stats["approx_kl"] += [approx_kl.item()]
                    stats["clip_frac"] += [np.mean(clip_fracs)]

                    # Take gradient step.
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            self.scheduler.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
            print(f"Training episode {episode}")
            print(f"Episodic Return: {np.mean(total_episodic_return)}")
            print(f"Episode Length: {end_step}")
            print("")

            # Store episode statistics.
            stats["episodic_return"] += [np.mean(total_episodic_return)]
            stats["episodic_length"] += [end_step]

        return stats


    def evaluate(self, render=False, max_cycles=None, max_episodes=1, seed=None):
        ''' Evaluates the policy. '''

        self.learner.eval()

        if max_cycles != None:
            self.env.max_cycles = max_cycles

        with torch.no_grad():
            # render max_episodes episodes out
            for episode in range(max_episodes):
                obs, info = self.env.reset(seed=seed)
                if render:
                    clear_output(wait=True)
                    self.env.render()
                obs = self.learner.batchify_obs(self.env.observation_space(self.env.possible_agents[0]), obs, self.device)
                terms = [False]
                truncs = [False]
                while not any(terms) and not any(truncs):
                    actions, logprobs, _, values = self.learner.get_action_and_value(obs)
                    obs, rewards, terms, truncs, infos = self.env.step(self.learner.unbatchify(actions, self.env))
                    obs = self.learner.batchify_obs(self.env.observation_space(self.env.possible_agents[0]), obs, self.device)
                    terms = [terms[a] for a in terms]
                    truncs = [truncs[a] for a in truncs]
                    if render:
                        clear_output(wait=True)
                        self.env.render()


class PPONetwork(nn.Module):
    def __init__(self, num_actions, num_agents, observation_size, device):
        super().__init__()

        self.num_actions = num_actions
        self.num_agents = num_agents
        self.device = device


        self.network = nn.Sequential(
            # self._layer_init(nn.Conv2d(4, 32, 3, padding=1)),
            # nn.MaxPool2d(2),
            # nn.ReLU(),
            # self._layer_init(nn.Conv2d(32, 64, 3, padding=1)),
            # nn.MaxPool2d(2),
            # nn.ReLU(),
            # self._layer_init(nn.Conv2d(64, 128, 3, padding=1)),
            # nn.MaxPool2d(2),
            # nn.ReLU(),
            # nn.Flatten(),
            self._layer_init(nn.Linear(observation_size, 512)),
            nn.ReLU(),      
            self._layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(512, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(512, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        logits_split = torch.split(logits, split_size_or_sections=1, dim=0)

        if action is None:
            action = []
            for i in range(self.num_agents):
                probs = Categorical(logits = logits_split[i])
                action.append(probs.sample().item())
            action = torch.tensor(action)
        action = action.to(self.device)

        entropy = []
        probs = Categorical(logits = logits)
        logprobs = probs.log_prob(action)
        entropy = probs.entropy()
        # logprobs = torch.tensor(logprobs).to(self.device)
        # entropy = torch.tensor(entropy).to(self.device)
        return action, logprobs, entropy, self.critic(hidden)

        # logprobs = []
        # entropy = []
        # for i in range(self.num_agents):
        #     probs = Categorical(logits = logits_split[i])
        #     logprobs.append(probs.log_prob(action[i]).item())
        #     entropy.append(probs.entropy().item())
        # logprobs = torch.tensor(logprobs).to(self.device)
        # entropy = torch.tensor(entropy).to(self.device)
        # return action, logprobs, entropy, self.critic(hidden)


    def batchify_obs(self, obs_space, obs, device):
        """Converts PZ style observations to batch of torch arrays."""
        # convert to list of np arrays

        #np.stack is a method that concencate the tensors along the new axis
        obs = np.stack([flatten(obs_space, obs[a]) for a in obs], axis=0)
        # convert to torch
        obs = torch.tensor(obs).to(device)

        return obs


    def batchify(self, x, device):
        """Converts PZ style returns to batch of torch arrays."""
        # convert to list of np arrays
        x = np.stack([x[a] for a in x], axis=0)
        # convert to torch
        x = torch.tensor(x).to(device)

        return x


    def unbatchify(self, x, env):
        """Converts np array to PZ style arguments."""
        x = x.cpu().numpy()
        x = {a: x[i] for i, a in enumerate(env.possible_agents)}

        return x