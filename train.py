import os
import sys
sys.path.append('.')

import argparse
import pickle
import random
import numpy as np
import torch
import gym
import d4rl
import neorl

from models.temporal import TrajCondUnet
from models.diffusion import GaussianDiffusion
from models.score_model import ScoreModel
from models.actor import DeterministicActor
from data import BlockRankingDataset, ActorDataset
from utils.normalizer import DatasetNormalizer
from utils.logger import Logger, make_log_dirs
from utils.trainer import Trainer
from utils.timer import Timer
from utils.evaluator import Evaluator
from utils.helpers import make_dataset
from utils.filter import *
from wrappers.neorl_env import make_env

from gym.envs import register
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS

for env_name, env_cls in ALL_V2_ENVIRONMENTS.items():
    ID = f"mw_{env_name}"
    register(id=ID, entry_point="wrappers.metaworld:SawyerEnv", kwargs={"env_name": env_name})
    
    
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="FTB")
    parser.add_argument("--domain", type=str, default="d4rl")
    parser.add_argument("--task", type=str, default="hopper-medium-replay-v2")
    parser.add_argument("--data-type", type=str, default="low")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episode-len", type=int, default=1000)
    parser.add_argument("--discount", type=float, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # preference
    parser.add_argument("--use-human-label", type=bool, default=False)
    parser.add_argument("--pref-episode-len", type=int, default=100)
    parser.add_argument("--pref-num", type=int, default=15)
    parser.add_argument("--survival-reward", type=bool, default=False)
    parser.add_argument("--pref-embed-dim", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--pref-lr", type=float, default=1e-4)
    parser.add_argument("--pref-max-iters", type=int, default=10)
    parser.add_argument("--pref-num-steps-per-iter", type=int, default=100)
    parser.add_argument("--pref-batch-size", type=int, default=256)

    # diffusion
    parser.add_argument("--improve-step", type=int, default=20)
    parser.add_argument("--diff-episode-len", type=int, default=1000)
    parser.add_argument("--diff-embed-dim", type=int, default=128)
    parser.add_argument("--dim-mults", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--n-diffusion-steps", type=int, default=1000)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--diff-ema-start-epoch", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--diff-lr", type=float, default=1e-4)
    parser.add_argument("--diff-max-iters", type=int, default=500)
    parser.add_argument("--diff-num-steps-per-iter", type=int, default=1000)
    parser.add_argument("--diff-batch-size", type=int, default=50)

    # actor
    parser.add_argument("--actor-embed-dim", type=int, default=256)
    parser.add_argument("--actor-hidden-layer", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--actor-type", type=str, default="deterministic")
    parser.add_argument("--select-num", type=int, default=10)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=1.05)
    parser.add_argument("--actor-max-iters", type=float, default=100)
    parser.add_argument("--actor-num-steps-per-iter", type=float, default=1000)
    parser.add_argument("--actor-batch-size", type=int, default=256)

    return parser.parse_args()


def train(args=get_args()):
    # set seed everywhere
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # create env & load dataset
    if args.domain == "d4rl":
        env = gym.make(args.task)
        dataset = env.get_dataset()
        with open(os.path.join("dataset", "human_label", args.domain, f"{args.task}.pkl"), 'rb') as f:
            label_dataset = pickle.load(f)
        normalizer = DatasetNormalizer(dataset)
    elif args.domain == "metaworld":
        with open(os.path.join("dataset", "metaworld_dataset", f"{args.task}.pkl"), 'rb') as f:
            dataset = pickle.load(f)
        with open(os.path.join("dataset", "human_label", args.domain, f"{args.task}.pkl"), 'rb') as f:
            label_dataset = pickle.load(f)
        normalizer = DatasetNormalizer(dataset)
        select_dim = np.where(normalizer.normalizers["observations"].stds > 1e-4)[0]
        dataset["observations"] = dataset["observations"][..., select_dim]
        label_dataset["observations"] = label_dataset["observations"][..., select_dim]
        label_dataset["observations_2"] = label_dataset["observations_2"][..., select_dim]
        normalizer.normalizers["observations"].means = normalizer.normalizers["observations"].means[select_dim]
        normalizer.normalizers["observations"].stds = normalizer.normalizers["observations"].stds[select_dim]
        env = gym.make("mw_" + args.task, select_dim=select_dim)
    elif args.domain == "neorl":
        env = make_env(args.task)
        dataset= env.get_dataset(data_type=args.data_type, train_num=1000)
        with open(os.path.join("dataset", "human_label", args.domain, f"{args.task}-{args.data_type}.pkl"), 'rb') as f:
            label_dataset = pickle.load(f)
        normalizer = DatasetNormalizer(dataset)
    else:
        raise ValueError

    dataset = BlockRankingDataset(args.task, dataset, normalizer, label_dataset, args.episode_len, args.pref_num, args.device)

    args.obs_shape = env.observation_space.shape
    args.obs_dim = int(np.prod(args.obs_shape))
    args.action_dim = int(np.prod(env.action_space.shape))
    args.max_action = env.action_space.high[0]

    # create preference model
    preference_model = ScoreModel(observation_dim=args.obs_dim,
                                  action_dim=args.action_dim,
                                  survival_reward=args.survival_reward,
                                  device=args.device)
    preference_model.to(args.device)
    preference_optim = torch.optim.Adam(preference_model.parameters(), lr=args.pref_lr)
    preference_scheduler = torch.optim.lr_scheduler.LambdaLR(
        preference_optim,
        lambda steps: min((steps+1)/args.warmup_steps, 1)
    )

    # crearte diffusion model
    temporal_model = TrajCondUnet(args.diff_episode_len, args.obs_dim + args.action_dim, hidden_dim=args.diff_embed_dim, dim_mults=args.dim_mults, condition_dropout=args.dropout)
    diffusion_model = GaussianDiffusion( 
        model=temporal_model,
        horizon=args.diff_episode_len,
        observation_dim=args.obs_dim,
        action_dim=args.action_dim,
        n_timesteps=args.n_diffusion_steps,
        guidance_scale=args.guidance_scale,
        loss_type='l2',
        clip_denoised=False,
    )
    diffusion_model.to(args.device)
    diffusion_optim = torch.optim.Adam(diffusion_model.parameters(), args.diff_lr)

    # create actor model
    actor = DeterministicActor(
        observation_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_dim=args.actor_embed_dim,
        hidden_layer=args.actor_hidden_layer
    )
    actor.to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), args.actor_lr, weight_decay=args.weight_decay)
    
    # logger
    log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # key: output file name, value: output handler type
    output_config = {
        "consoleout_backup": "stdout",
        "critic_training_progress": "csv",
        "diffusion_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))
    
    # timer
    timer = Timer()
    # evaluator 
    evaluator = Evaluator(env, normalizer)
    # trainer
    trainer = Trainer(preference_model, diffusion_model, actor, dataset, logger, timer, evaluator, device=args.device)
    
    print('-------------train preference model-------------')
    trainer.train_preference(
        preference_optim,
        preference_scheduler,
        args.pref_max_iters,
        args.pref_num_steps_per_iter,
        batch_size=args.pref_batch_size,
    )

    preference_model.eval()
    dataset.set_returns(preference_model, discount=args.discount)
    dataset.block_ranking(args.improve_step)

    print('-------------train diffusion model-------------')
    trainer.train_diffusion(
        optim=diffusion_optim,
        ema_decay=0.995,
        epoch_start_ema=args.diff_ema_start_epoch,
        update_ema_every=10,
        max_iters=args.diff_max_iters,
        num_steps_per_iter=args.diff_num_steps_per_iter,
        batch_size=args.diff_batch_size,
    )
    
    print('-------------generate data-------------')
    generate_trajs, tlens = flow_to_better(preference_model, diffusion_model, dataset, select_num=args.select_num, threshold=args.threshold)

    bc_trajs = []
    for i in range(len(tlens)):
        bc_trajs.append(generate_trajs[i][:tlens[i]])
    bc_trajs = np.concatenate(bc_trajs)
    bc_dataset = dict()
    bc_dataset["observations"] = normalizer.unnormalize(bc_trajs[..., :args.obs_dim], "observations").reshape(-1, args.obs_dim)
    bc_dataset["actions"] = normalizer.unnormalize(bc_trajs[..., args.obs_dim:], "actions").reshape(-1, args.action_dim)

    bc_dataset = ActorDataset(bc_dataset, normalizer, device=args.device)

    print('-------------train actor-------------')
    trainer.train_actor(
        dataset=bc_dataset,
        optim=actor_optim,
        max_iters=args.actor_max_iters,
        num_steps_per_iter=args.actor_num_steps_per_iter,
        batch_size=args.actor_batch_size,
    ) 


if __name__ == '__main__':
    train()