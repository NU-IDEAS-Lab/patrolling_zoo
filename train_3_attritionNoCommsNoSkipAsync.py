from onpolicy.scripts.train.train_patrolling import get_config, parse_args, main

import os
os.environ["WANDB__SERVICE_WAIT"] = "300"

parser = get_config()
all_args = parse_args([], parser)



all_args.experiment_name = "3SPattritionNoCommsNoSkipAsyncAdjacency"
all_args.env_name = "Patrolling"
all_args.user_name = "ideas-mas"

all_args.num_agents = 6
all_args.agent_speed = 1.0
all_args.communication_model = "bernoulli"
all_args.communication_probability = 0.0
all_args.observe_method = "adjacency"
all_args.observe_method_global = "adjacency"
all_args.observe_bitmap_size = 40
all_args.observation_radius = 200
all_args.attrition_method = "none"
all_args.attrition_random_probability = 0.0
all_args.attrition_min_agents = 5
all_args.alpha = 1.0
all_args.beta = 0.5
all_args.reward_method_terminal = "average"
all_args.share_reward = False

all_args.graph_name = "cumberland"
all_args.graph_file = f"./patrolling_zoo/patrolling_zoo/env/{all_args.graph_name}.graph"
all_args.num_env_steps = 10e5 * 1 #total number of steps
all_args.episode_length = 400 #number of steps in a training episode
all_args.max_cycles = all_args.episode_length #number of steps in an environment episode

all_args.algorithm_name = "rmappo"
all_args.use_recurrent_policy = True
all_args.use_naive_recurrent_policy = False
all_args.use_centralized_V = True
all_args.use_gae = False
all_args.use_gae_amadm = True
all_args.share_policy = False
all_args.skip_steps_sync = False
all_args.skip_steps_async = True

all_args.save_interval = 10
all_args.n_rollout_threads = 10
all_args.cuda = True
all_args.cuda_idx = 2

all_args.use_wandb = True



main([], parsed_args = all_args)