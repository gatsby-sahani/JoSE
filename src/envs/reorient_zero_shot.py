import os
import joblib
import numpy as np

from src.envs.env_factory import make_env, make_vectorized_env
from src.agents.policy_factory import load_trained_policy

from tqdm import tqdm
import gc
import torch
import wandb


def get_env_and_model(cfg, ica_pca):

    env = make_vectorized_env(
        cfg,
        make_env,
        ica_pca,
        cfg.trainer.num_training_envs,
    )

    model, vec = load_trained_policy(cfg, env)

    return env, vec, model


def zero_shot_generalization(
    cfg, ica_pca, run_path, timestamp, wandb_run, episodes=1_000
):

    if cfg.agent.name == "sac_sar_state_independent":
        joblib.dump(ica_pca[0], os.path.join(run_path, "ica_demo.pkl"))
        joblib.dump(ica_pca[1], os.path.join(run_path, "pca_demo.pkl"))
        joblib.dump(ica_pca[2], os.path.join(run_path, "pca_descaler_demomo.pkl"))

    cfg.agent.load_dir = os.path.join(timestamp, wandb_run.id)
    cfg.agent.load_step = cfg.trainer.total_timesteps

    # ----------------------------
    # Examine zero-shot generalization for in-domain and out-of-domain objects
    # ----------------------------
    for env_id in ["myoHandReorientID-v0", "myoHandReorientOOD-v0"]:
        cfg.env.env_id = env_id

        wandb_run = wandb.init(
            project=env_id,
            config=dict(cfg),
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

        env, vec, model = get_env_and_model(cfg, ica_pca)

        rewards = []
        solved = []
        solved_frac = []

        obs = env.reset()

        ep_rewards = np.zeros(env.num_envs)
        ep_solved = np.zeros(env.num_envs)

        completed_episodes = 0
        pbar = tqdm(total=episodes)

        # ----------------------------
        # Rollout
        # ----------------------------
        while completed_episodes < episodes:
            obs = vec.normalize_obs(obs)
            actions, _ = model.predict(obs, deterministic=True)

            obs, r, dones, infos = env.step(actions)

            ep_rewards += r

            for i in range(env.num_envs):
                ep_solved[i] += infos[i].get("solved", 0)

                if dones[i]:
                    rewards.append(ep_rewards[i])
                    solved.append(ep_solved[i] > 0)

                    solved_frac.append(
                        ep_solved[i] / env.get_attr("spec")[0].max_episode_steps
                    )

                    ep_rewards[i] = 0.0
                    ep_solved[i] = 0.0

                    completed_episodes += 1
                    pbar.update(1)

                    if completed_episodes >= episodes:
                        break

        pbar.close()

        results = {
            "rewards": {
                "mean": np.mean(rewards),
                "sem": np.std(rewards, ddof=1) / np.sqrt(len(rewards)),
            },
            "solved": {
                "mean": np.mean(solved),
                "sem": np.std(solved, ddof=1) / np.sqrt(len(solved)),
            },
            "solved_frac": {
                "mean": np.mean(solved_frac),
                "sem": np.std(solved_frac, ddof=1) / np.sqrt(len(solved_frac)),
            },
        }

        print(results)

        wandb.log(results, step=1)
        wandb.finish()

        env.close()

        del env
        gc.collect()
        torch.cuda.empty_cache()
