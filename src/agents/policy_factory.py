import os
import numpy as np
import jax

from src.agents.josepi.sac import SAC as JoSEPi
from src.agents.play_phase_agent.sac import SAC as PlayPhaseAgent
from src.envs.wrappers import ReplayBufferWrapper
from src.agents.sar_state_dependent.sac import SAC as StateDependentSAR
from sbx.sac import SAC as SAC
from src.agents.action_squashing_fns import get_action_squasher
from stable_baselines3.common.vec_env import VecNormalize


def initialize_policy(cfg, env, wandb_run):

    # use float64 for numerical stability to calculate the correction for log prob on a manifold in JoSEPi
    jax.config.update("jax_enable_x64", cfg.agent.use_float64)

    policy_kwargs = {"net_arch": dict(cfg.agent.net_arch)}

    if cfg.agent.name == "josepi":
        model_class = JoSEPi

        controlled_variables = env.get_attr("controlled_variables")[0]
        model_kwargs = dict(
            controlled_variables=controlled_variables,
            pi_and_Q_observations=env.get_attr("pi_and_Q_observations")[0],
            dynamics_observations=env.get_attr("dynamics_observations")[0],
            action_squasher=get_action_squasher(env, cfg),
            target_entropy=(
                -np.float32(cfg.agent.optim.rank_k)
                if cfg.agent.optim.rank_k
                else -np.float32(len(controlled_variables))
            ),
        )

    elif cfg.agent.name == "play_phase_agent":
        model_class = PlayPhaseAgent
        model_kwargs = {
            "replay_buffer_class": ReplayBufferWrapper,
            "replay_buffer_kwargs": {
                "total_buffer_size": int(cfg.trainer.total_timesteps),
            },
        }
        policy_kwargs["pi_and_Q_observations"] = env.get_attr("pi_and_Q_observations")[
            0
        ]

    elif cfg.agent.name in ["sac", "sac_sar_state_independent"]:
        model_class = SAC
        model_kwargs = {}

    elif cfg.agent.name in ["sac_sar_state_dependent"]:
        model_class = StateDependentSAR
        wand_run_id = cfg.agent.load_dir.split("/")[-1]
        model_kwargs = {
            "pre_train_buffer_path": os.path.join(
                "play_phase_data", f"{wand_run_id}.pkl"
            )
        }

    model = model_class(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=os.path.dirname(wandb_run.dir),
        **cfg.agent.optim,
        **model_kwargs,
    )

    return model


def load_trained_policy(cfg, env, generate_sar=False):

    if generate_sar:
        model_class = PlayPhaseAgent  # PlayPhaseAgent was used in the play phase
    elif cfg.agent.name == "josepi":
        model_class = JoSEPi
    elif cfg.agent.name in ["sac", "sac_sar_state_independent"]:
        model_class = SAC
    elif cfg.agent.name == "sac_sar_state_dependent":
        model_class = StateDependentSAR

    run_path = os.path.join("checkpoints", cfg.agent.load_dir)
    steps_str = str(cfg.agent.load_step) + "_steps"
    model = model_class.load(os.path.join(run_path, "rl_model_" + steps_str), env=env)

    if cfg.env.norm_obs:
        vec = VecNormalize.load(
            os.path.join(run_path, f"rl_model_vecnormalize_{steps_str}.pkl"),
            env,
        )
        model.policy.vec_norm = vec
    else:
        vec = None

    return model, vec
