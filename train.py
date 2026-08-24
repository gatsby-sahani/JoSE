#!/usr/bin/env python3
"""Train a policy."""
import os
import hydra
import gc
import torch
import jax
import wandb

from src.envs.env_factory import get_env
from src.agents.policy_factory import initialize_policy
from src.callbacks.callback_factory import get_callback
from src.envs.reorient_zero_shot import zero_shot_generalization

os.environ.setdefault("MUJOCO_GL", "egl")

cwd = os.path.dirname(os.path.abspath(__file__))

print(jax.devices())


@hydra.main(
    version_base="1.3", config_path=os.path.join(cwd, "configs"), config_name="config"
)
def main(cfg: "DictConfig"):

    wandb_run = wandb.init(
        project=cfg.env.env_id,
        config=dict(cfg),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    env, eval_env, ica_pca = get_env(cfg)

    model = initialize_policy(cfg, env, wandb_run)

    callback, run_path, timestamp = get_callback(cfg, eval_env, wandb_run)

    model.learn(
        total_timesteps=cfg.trainer.total_timesteps,
        reset_num_timesteps=False,
        callback=callback,
    )

    wandb_run.finish()
    env.close()
    eval_env.close()

    # after training on myoHandReorient100-v0
    # evaluate zero-shot generalization on myoHandReorientID/OOD-v0
    if cfg.env.env_id == "myoHandReorient100-v0":
        model.policy.to("cpu")
        del model, env, eval_env
        gc.collect()
        torch.cuda.empty_cache()
        zero_shot_generalization(cfg, ica_pca, run_path, timestamp, wandb_run)


if __name__ == "__main__":
    main()
