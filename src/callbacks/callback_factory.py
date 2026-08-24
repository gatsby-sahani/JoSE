import os
from datetime import datetime

from stable_baselines3.common.callbacks import CheckpointCallback
from src.callbacks.callback import (
    TensorboardCallback,
    FallbackCheckpoint,
    EvalCallback,
)


def get_callback(cfg, eval_env, wandb_run):

    timestamp = datetime.now().strftime("%Y-%m-%d")
    run_path = os.path.join("checkpoints", timestamp, wandb_run.id)

    callback = [
        CheckpointCallback(
            save_freq=max(cfg.trainer.save_freq // cfg.trainer.num_training_envs, 1),
            save_path=run_path,
            name_prefix="rl_model",
            save_replay_buffer=cfg.trainer.save_replay_buffer,
            save_vecnormalize=cfg.trainer.save_vecnormalize,
        )
    ]

    callback += [TensorboardCallback()]
    callback += [
        FallbackCheckpoint(
            save_freq=max(
                cfg.trainer.fallback_freq // cfg.trainer.num_training_envs, 1
            ),
            save_path=run_path,
            name_prefix="rl_model",
            save_replay_buffer=cfg.trainer.save_replay_buffer,
            save_vecnormalize=cfg.trainer.save_vecnormalize,
        )
    ]

    callback += [
        EvalCallback(
            max(cfg.trainer.eval_freq // cfg.trainer.num_training_envs, 1), eval_env
        )
    ]

    return callback, run_path, timestamp
