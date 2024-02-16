from onpolicy.scripts.train.train_patrolling import get_config, parse_args, main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

parser = get_config()
all_args = parse_args([], parser)

all_args.experiment_name = "gnnMultiLayerNeighbors"
all_args.env_name = "Patrolling"
all_args.user_name = "ideas-mas"

all_args.num_agents = 6
all_args.agent_speed = 40.0
all_args.action_method = "neighbors"
all_args.observe_method = "pyg"
all_args.observe_method_global = "adjacency"
all_args.observation_radius = 200.0
all_args.observation_bitmap_size = 40
all_args.alpha = 0.0
all_args.beta = 1.0
# all_args.reward_method_terminal = "averageAverage"
all_args.reward_method_terminal = "none"
all_args.reward_interval = 1

all_args.graph_name = "23nodes"
all_args.graph_file = f"patrolling_zoo/env/{all_args.graph_name}.graph"
all_args.num_env_steps = 10e5 * 10 #total number of steps
all_args.episode_length = 150 #number of steps in a training episode
all_args.max_cycles = all_args.episode_length #number of steps in an environment episode

all_args.algorithm_name = "mappo"
all_args.use_gnn_policy = True
all_args.use_recurrent_policy = True
all_args.use_naive_recurrent_policy = False
all_args.use_centralized_V = True
all_args.use_gae = False
all_args.use_gae_amadm = True
all_args.share_policy = False
all_args.sep_share_policy = False
all_args.share_reward = False
all_args.skip_steps_sync = False
all_args.skip_steps_async = True
all_args.use_ReLU = False
# all_args.lr = 1e-4
# all_args.entropy_coef = 0.1
all_args.hidden_size = 512

all_args.n_rollout_threads = 1
all_args.save_interval = 10
all_args.cuda = True
all_args.cuda_idx = 7

all_args.use_wandb = True



main([], parsed_args = all_args)