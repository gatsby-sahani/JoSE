from myosuite.utils import gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from src.envs.vecmonitor import VecMonitor
from src.agents.controlled_variables import get_controlled_variables
from src.envs.wrappers import (
    RenderWrapper,
    SparseRewardWrapper,
    ActInfoWrapper,
)
from src.agents.sar_state_independent.utils import (
    get_sar_data,
    get_state_independent_sar,
    SynergyWrapper,
    SynNoSynWrapper,
)


def make_env(cfg, controlled_variable_wrapper, run, ica_pca):
    """
    General make environment function.
    """

    def _init():
        env = gym.make(
            cfg.env.env_id,
            normalize_act=cfg.agent.norm_act,
            **cfg.env.env_kwargs,
        )
        if cfg.agent.name in ["josepi", "play_phase_agent"]:
            env = controlled_variable_wrapper(env, cfg)
        if cfg.env.reward_type == "sparse":
            env = SparseRewardWrapper(env)
        if cfg.agent.name == "sac_sar_state_independent":
            if cfg.agent.synnosyn:
                env = SynNoSynWrapper(env, ica_pca)
            else:
                env = SynergyWrapper(env, ica_pca)
        env = RenderWrapper(env, cfg)
        if "Adroit" not in cfg.env.env_id:
            env = ActInfoWrapper(env)
        env.unwrapped.seed(run)
        return env

    set_random_seed(cfg.env.seed)
    return _init


def make_sar_env(cfg, controlled_variable_wrapper, run, ica_pca):
    """
    Make environment function for generating SAR.
    """

    def _init():
        env = gym.make(
            "myoHandReorient8-v0",
            normalize_act=cfg.agent.norm_act,
            **cfg.env.env_kwargs,
        )  # reload the play-phase environment - myoHandReorient8-v0
        env = controlled_variable_wrapper(
            env, cfg
        )  # the controlled_variable_wrapper was used in the play phase, so reapply it
        env = ActInfoWrapper(env)
        env.unwrapped.seed(run)
        return env

    set_random_seed(cfg.env.seed)
    return _init


def make_vectorized_env(cfg, env_fn, ica_pca=None, num_envs=1):
    controlled_variable_wrapper = get_controlled_variables(cfg)
    start_index = cfg.env.seed * cfg.trainer.num_training_envs
    env = SubprocVecEnv(
        [
            env_fn(
                cfg,
                controlled_variable_wrapper,
                start_index + i,
                ica_pca,
            )
            for i in range(num_envs)
        ],
        start_method="forkserver",
    )
    return env


def get_env(cfg):

    if cfg.agent.name in ["sac_sar_state_independent", "sac_sar_state_dependent"]:
        acts = get_sar_data(cfg)

    ica_pca = (
        get_state_independent_sar(acts)
        if cfg.agent.name == "sac_sar_state_independent"
        else None
    )

    # ----------------------------
    # Training env
    # ----------------------------
    env = make_vectorized_env(
        cfg,
        make_env,
        ica_pca,
        cfg.trainer.num_training_envs,
    )

    env = VecMonitor(env, max_episode_steps=cfg.env.env_kwargs.max_episode_steps)

    if cfg.env.norm_obs:
        env = VecNormalize(
            env,
            norm_obs=cfg.env.norm_obs,
            norm_reward=cfg.env.norm_reward,
            gamma=cfg.agent.optim.gamma,
            clip_obs=cfg.env.clip_obs,
        )

    # ----------------------------
    # Evaluation env
    # ----------------------------
    eval_env = make_vectorized_env(
        cfg,
        make_env,
        ica_pca,
    )
    eval_env = VecMonitor(
        eval_env, max_episode_steps=cfg.env.env_kwargs.max_episode_steps
    )
    eval_env = VecNormalize(eval_env) if cfg.env.norm_obs else eval_env

    return env, eval_env, ica_pca
