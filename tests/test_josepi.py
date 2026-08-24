import numpy as np


def test_josepi_instantiate_and_predict():
    from hydra import compose, initialize
    from omegaconf import OmegaConf
    from stable_baselines3.common.vec_env import DummyVecEnv

    import src.envs  # register environments
    from myosuite.utils import gym
    from src.agents.action_squashing_fns import get_action_squasher
    from src.agents.controlled_variables import ObsWrapperQvel
    from src.agents.josepi.sac import SAC as JoSEPi

    with initialize(config_path="../configs", version_base=None):
        cfg = compose("config", overrides=["env=myoHandReorient8-v0", "agent=josepi"])

    def _make():
        env = gym.make(
            cfg.env.env_id, normalize_act=cfg.agent.norm_act, **cfg.env.env_kwargs
        )
        env = ObsWrapperQvel(env, cfg)
        return env

    env = DummyVecEnv([_make])
    controlled_variables = env.get_attr("controlled_variables")[0]
    optim = OmegaConf.to_container(cfg.agent.optim, resolve=True)

    model = JoSEPi(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": dict(cfg.agent.net_arch)},
        verbose=0,
        controlled_variables=controlled_variables,
        pi_and_Q_observations=env.get_attr("pi_and_Q_observations")[0],
        dynamics_observations=env.get_attr("dynamics_observations")[0],
        action_squasher=get_action_squasher(env, cfg),
        target_entropy=(
            -np.float32(optim["rank_k"])
            if optim["rank_k"]
            else -np.float32(len(controlled_variables))
        ),
        **optim,
    )

    obs = env.reset()
    obs_mean = np.zeros(obs.shape[-1], dtype=np.float32)
    obs_std = np.ones(obs.shape[-1], dtype=np.float32)
    actions, _ = model.policy._predict(obs, obs_mean, obs_std, deterministic=True)
    assert actions.shape == (1, env.action_space.shape[0])
