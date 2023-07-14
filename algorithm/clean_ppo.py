from gymnasium.spaces.utils import flatten, flatten_space
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import numpy as np
import time
from IPython.display import clear_output

from torch.utils.tensorboard import SummaryWriter



from .algorithm import BaseAlgorithm

class PPO(BaseAlgorithm):
    ''' This algorithm implements PPO for the patrolling problem.
        Adapted from an example by Jet (https://github.com/jjshoots). '''

    def __init__(self, env,
            env_id = "patrolling-v0",
            track = False,
            wandb_project_name = "CleanRL",
            wandb_entity = None,
            ent_coef = 0.01,
            vf_coef = 0.5,
            clip_coef = 0.2,
            gamma = 0.99,
            learning_rate = 0.0001,
            frame_size = (64, 64),
            num_steps = 128,
            total_timesteps = 500000,
            gae_lambda = 0.95,
            update_epochs = 4,
            anneal_lr = True,
            num_minibatches = 4,
            norm_adv = True,
            clip_vloss = True,
            max_grad_norm = 0.5,
            target_kl = None,
        ):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(env, device)

        self.env_id = env_id
        self.track = track
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        
        self.num_env = 1
    
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.clip_coef = clip_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.learning_rate = learning_rate
        self.update_epochs = update_epochs
        self.frame_size = frame_size
        self.num_steps = num_steps
        self.total_timesteps = total_timesteps
        self.anneal_lr = anneal_lr
        self.batch_size = int(self.num_env * self.num_steps)
        self.num_minibatches = num_minibatches
        self.minibatch_size =  int(self.batch_size // self.num_minibatches)
        self.norm_adv = norm_adv
        self.clip_vloss = clip_vloss

        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl

        self.num_agents = len(env.possible_agents)

    
    def train(self):

        run_name = f"{self.env_id}__{int(time.time())}"

        if self.track:
            import wandb

            wandb.init(
                project=self.wandb_project_name,
                entity=self.wandb_entity,
                sync_tensorboard=True,
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )


        writer = SummaryWriter(f"runs/{run_name}")
        
        env = self.env
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        agent = Agent(env).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((self.num_steps,)+ env.observation_space.shape).to(device)
        actions = torch.zeros((self.num_steps, self.num_agents)).to(device)
        logprobs = torch.zeros((self.num_steps, self.num_agents,)).to(device)
        rewards = torch.zeros((self.num_steps, self.num_agents,)).to(device)
        dones = torch.zeros((self.num_steps, self.num_agents,)).to(device)
        values = torch.zeros((self.num_steps, self.num_agents,)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs = torch.Tensor(self.env.reset()).to(device)
        next_done = torch.zeros(self.num_env).to(device)
        num_updates = self.total_timesteps // self.batch_size

        for update in range(1, num_updates + 1):    
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * self.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                global_step += 1 * self.num_env
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob
                
                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, done, info = self.env.step(action.cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break
                            
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.env.observation_space.shape)
            b_logprobs = logprobs.reshape(-1, self.num_agents)
            b_actions = actions.reshape(-1, self.num_agents)
            b_advantages = advantages.reshape(-1, self.num_agents)
            b_returns = returns.reshape(-1, self.num_agents)
            b_values = values.reshape(-1, self.num_agents)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1,3)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), self.max_grad_norm)
                    optimizer.step()

                if self.target_kl is not None:
                    if approx_kl > self.target_kl:
                        break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            #print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
            writer.add_scalar("charts/episodic_return", torch.sum(rewards), global_step)
            self.agent = agent

            print("Update: %s, total reward: %s, timestep: %s"%(update, int(torch.sum(rewards)), global_step))

        env.close()
        writer.close()

        return agent


    def evaluate(self, agent, render=False, max_cycles=None):
        ''' Evaluates the policy. '''

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if max_cycles != None:
            self.env.maxSteps = max_cycles

        with torch.no_grad():
            # render 2 episodes out
            for episode in range(2):
                obs = torch.Tensor(self.env.reset()).to(device)
                if render:
                    clear_output(wait=True)
                    self.env.render()
                terms = [False]
                truncs = [False]
                while not any(terms) and not any(truncs):
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    obs, reward, done, info = self.env.step(actions.cpu().numpy())
                    obs = torch.Tensor(obs).to(device)
                    if render:
                        clear_output(wait=True)
                        self.env.render()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def serialize_obs(obs):
    agent_states = torch.tensor([list(val) for val in obs['agent_state'].values()])
    vertex_states = torch.tensor([val for val in obs['vertex_state'].values()])
    return torch.cat((agent_states.view(-1), vertex_states)).float()

class Agent(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 4096)),
            nn.ReLU(),
            layer_init(nn.Linear(4096, 4096)),
            nn.ReLU(),
            layer_init(nn.Linear(4096, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 3), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 2048)),
            nn.ReLU(),
            layer_init(nn.Linear(2048, 1024)),
            nn.ReLU(),
            layer_init(nn.Linear(1024, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, env.action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        logits = logits.view(-1, 3, 40)  # Reshape logits to size n*3*40
        n = logits.shape[0] # get the batch size
        if action is None:
            action = Categorical(logits=logits).sample()
        else:
            action = action.view(n, -1)  # Reshape the action tensor if provided
        logprobs = []
        for i in range(n): # loop over the batch dimension
            single_action = action[i] # get a single action of shape (3,)
            single_logits = logits[i] # get the corresponding logits
            total_probs = Categorical(logits=single_logits)
            logprob = total_probs.log_prob(single_action)
            logprobs.append(logprob)
        # After the loop, you might want to stack the list of tensors into a single tensor:
        logprobs = torch.stack(logprobs)
        entropy = Categorical(logits=logits).entropy().mean()  # Get mean entropy over the batch
        value = self.critic(x)
        return action, logprobs, entropy, value



        logprobs = []
        for i in range(action.shape[0]): # loop over the batch dimension
            single_action = action[i] # get a single action of shape (3,)
            single_logits = logits[i] # get the corresponding logits
            total_probs = torch.distributions.Categorical(logits=single_logits)
            logprob = total_probs.log_prob(single_action)
            logprobs.append(logprob)
            
        # After the loop, you might want to stack the list of tensors into a single tensor:
        logprobs = torch.stack(logprobs)

        return action, logprobs, total_probs.entropy(), self.critic(hidden)


