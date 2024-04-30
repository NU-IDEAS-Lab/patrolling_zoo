#!/usr/bin/env python
# python standard libraries
import os
from pathlib import Path
import sys
import socket

# third-party packages
import numpy as np
import setproctitle
import torch
import wandb

# code repository sub-packages
from onpolicy.config import get_config
from onpolicy.envs.patrolling.SDEnv import SDEnv
from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "search-deliver":
                env = SDEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Football":
                env = SDEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Weight of local reward.")
    parser.add_argument("--beta", type=float, default=1000.0,
                        help="Weight of global reward.")
    parser.add_argument("--skip_steps_async", type=bool, default=False,
                        help="Whether to skip steps with no action required (by any agent).")
    parser.add_argument("--skip_steps_sync", type=bool, default=False,
                        help="Whether to skip steps with no action required (by any agent).")
    parser.add_argument("--graph_file", type=str,
                        default="", 
                        help="The path to the graph file.")
    parser.add_argument("--graph_name", type=str,
                        default="4nodes", 
                        help="which graph to run on.")
    parser.add_argument("--graph_random", type=bool, default=False,
                        help="Whether to use a random graph.")
    parser.add_argument("--graph_random_nodes", type=int,
                        default=40,
                        help="The number of random nodes to generate.")
    parser.add_argument("--max_cycles", type=int, default=1000,
                        help="max number of cycles for the environment.")
    parser.add_argument("--reward_method_terminal", type=str,
                        default="average", 
                        help="the method to use for terminal reward.")
    parser.add_argument("--reward_interval", type=int, default=-1,
                        help="number of steps between the periodic reward. -1 disables periodic reward")
    parser.add_argument("--num_agents", type=int, default=3,
                        help="number of controlled players.")
    parser.add_argument("--agent_speed", type=float, default=10.0,
                        help="the speed of each agent")
    parser.add_argument("--representation", type=str, default="simple115v2", 
                        choices=["simple115v2", "extracted", "pixels_gray", 
                                 "pixels"],
                        help="representation used to build the observation.")
    parser.add_argument("--rewards", type=str, default="scoring", 
                        help="comma separated list of rewards to be added.")
    parser.add_argument("--action_method", type=str, default="full", 
                        help="the action method to use")
    parser.add_argument("--observe_method", type=str, default="adjacency", 
                        help="the observation method to use")
    parser.add_argument("--observe_method_global", type=str, default=None, 
                        help="the observation method to use for global observation")
    parser.add_argument("--observe_bitmap_size", type=int, default=50, 
                        help="the size (squared) to which the bitmap should be scaled for observation")
    parser.add_argument("--observation_radius", type=float, default=np.Inf, 
                        help="the observable radius for each agent")
    parser.add_argument("--attrition_method", type=str, default="none", 
                        help="the method to use for agent attrition")
    parser.add_argument("--attrition_fixed_times", type=list, default=[], 
                        help="the fixed attrition times")
    parser.add_argument("--attrition_random_probability", type=float, default=0.0,
                        help="the random attrition probability")
    parser.add_argument("--attrition_min_agents", type=int, default=2,
                        help="the minimum number of agents that must be present for attrition to occur")
    parser.add_argument("--communication_model", type=str, default="none", 
                        help="the model name to use for communication. The \"none\" model indicates no comms allowed")
    parser.add_argument("--communication_probability", type=float, default=0.1, 
                        help="the probability of successful communication")
    parser.add_argument("--sep_share_policy", type=bool, default=True, 
                        help="Whether to share the policy amongst agents (even though using the separated runner)")
    parser.add_argument("--smm_width", type=int, default=96,
                        help="width of super minimap.")
    parser.add_argument("--smm_height", type=int, default=72,
                        help="height of super minimap.")
    parser.add_argument("--remove_redundancy", action="store_true", 
                        default=False, 
                        help="by default False. If True, remove redundancy features")
    parser.add_argument("--zero_feature", action="store_true", 
                        default=False, 
                        help="by default False. If True, replace -1 by 0")
    parser.add_argument("--eval_deterministic", action="store_false", 
                        default=True, 
                        help="by default True. If False, sample action according to probability")
    parser.add_argument("--share_reward", action='store_false', 
                        default=True, 
                        help="by default true. If false, use different reward for each agent.")

    parser.add_argument("--save_videos", action="store_true", default=False, 
                        help="by default, do not save render video. If set, save video.")
    parser.add_argument("--video_dir", type=str, default="", 
                        help="directory to save videos.")
    parser.add_argument("--cuda_idx", type=int, default=0, 
                        help="Index of the GPU to use")
                        
    all_args = parser.parse_known_args(args)[0]

    return all_args


def validateArgs(all_args):
    if all_args.graph_random:
        all_args.graph_name = f"random{all_args.graph_random_nodes}"

    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False. Note that GRF is a fully observed game, so ippo is rmappo.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError
    
    if all_args.skip_steps_async and all_args.skip_steps_sync:
        raise ValueError("Cannot skip steps in both async and sync mode.")


def main(args, parsed_args=None):
    if parsed_args is None:
        parser = get_config()
        all_args = parse_args(args, parser)
    else:
        all_args = parsed_args
    validateArgs(all_args)

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(f"cuda:{all_args.cuda_idx}")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.graph_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # get date and time in format 20230816-150432
    import datetime
    date_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                            project=all_args.env_name,
                            entity=all_args.user_name,
                            notes=socket.gethostname(),
                            name="-".join([
                                all_args.algorithm_name,
                                all_args.experiment_name,
                                str(date_time),
                                "seed" + str(all_args.seed)
                            ]),
                            group=all_args.graph_name,
                            dir=str(run_dir),
                            job_type="training",
                            reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle("-".join([
        all_args.env_name, 
        all_args.graph_name, 
        all_args.algorithm_name, 
        all_args.experiment_name
    ]) + "@" + all_args.user_name)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.patrolling_runner import PatrollingRunner as Runner
    else:
        from onpolicy.runner.separated.patrolling_runner import PatrollingRunner as Runner

    try:
        runner = Runner(config)
        runner.run()
    except KeyboardInterrupt:
        wandb.finish(exit_code=1)
        print("wandb due to keyboard interrupt")

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
