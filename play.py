#!/usr/bin/env python3
"""Load a trained JoSEPi policy from HuggingFace."""

import src.envs  # registers custom environments

import argparse
import glob
import os
from huggingface_hub import snapshot_download
from myosuite.utils import gym
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.agents.josepi.sac import SAC as JoSEPi
from src.agents.controlled_variables import ObsWrapperQvel

REPO_ID = "jamesheald/joint-space-empowerment"


def load_and_play(hand, task, seed, repo_id):
    task_folder = f"{hand}/{task}"
    seed_folder = f"{task_folder}/seed_{seed}"
    yaml_folder = os.path.dirname(task_folder) if "/" in task else task_folder

    # Download config, model, and normalization stats
    local = snapshot_download(
        repo_id, allow_patterns=[f"{yaml_folder}/*.yaml", f"{seed_folder}/*"]
    )
    config_path = glob.glob(f"{local}/{yaml_folder}/*.yaml")[0]
    model_path = glob.glob(f"{local}/{seed_folder}/*.zip")[0]
    stats_path = glob.glob(f"{local}/{seed_folder}/*.pkl")[0]

    # Load environment config
    cfg = OmegaConf.create({"env": OmegaConf.load(config_path)})
    env_kwargs = OmegaConf.to_container(cfg.env.env_kwargs, resolve=True)

    env_id = cfg.env.env_id

    def make_init(run_idx):
        def _init():
            e = gym.make(env_id, normalize_act=True, **env_kwargs)
            e.unwrapped.seed(run_idx)
            return ObsWrapperQvel(e, cfg)

        return _init

    raw_env = DummyVecEnv([make_init(0)])

    # Apply normalization stats
    env = VecNormalize.load(stats_path, raw_env)
    env.training = False
    env.norm_reward = False

    # Load policy
    model = JoSEPi.load(model_path, env=raw_env)
    model.policy.vec_norm = env

    obs = env.reset()

    while True:
        action, _ = model.predict(obs)
        obs, _, done, _ = env.step(action)

        env.render()

        if done:
            obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, choices=["Adroit", "MyoHand"])
    parser.add_argument(
        "--task",
        required=True,
        choices=[
            "BaodingBalls",
            "DieReorient",
            "KeyTurn",
            "PenTwirl",
            "Reorient8-sparse",
            "Reorient100/Training",
        ],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--repo-id", default=REPO_ID)
    args = parser.parse_args()

    load_and_play(args.hand, args.task, args.seed, args.repo_id)
