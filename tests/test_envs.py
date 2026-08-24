import pathlib

import pytest
from hydra import compose, initialize
from omegaconf import OmegaConf

import src.envs  # registers all custom environments
from myosuite.utils import gym

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_ADROIT_AVAILABLE = (_PROJECT_ROOT / "external_deps" / "Adroit").exists()


def _env_kwargs(env_id):
    """Load env_kwargs from the training config."""
    with initialize(config_path="../configs", version_base=None):
        cfg = compose("config", overrides=[f"env={env_id}"])
    return OmegaConf.to_container(cfg.env.env_kwargs, resolve=True)


MYOHAND_ENVS = [
    "CustomMyoChallengeBaodingP1-v1",
    "CustomMyoChallengeDieReorientP2-v0",
    "myoHandKeyTurnRandom-v0",
    "myoHandPenTwirlRandom-v0",
    "myoHandReorient8-v0",
    "myoHandReorient100-v0",
    "myoHandReorientID-v0",
    "myoHandReorientOOD-v0",
]

ADROIT_ENVS = [
    "CustomAdroitBaodingP1-v1",
    "CustomAdroitDieReorientP2-v0",
    "CustomAdroitKeyTurnRandom-v0",
    "CustomAdroitPenTwirlRandom-v0",
]


def _run_env(env_id):
    env = gym.make(env_id, **_env_kwargs(env_id))
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    action = env.action_space.sample()
    obs, reward, *_ = env.step(action)
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    env.close()


@pytest.mark.parametrize("env_id", MYOHAND_ENVS)
def test_myohand_env(env_id):
    _run_env(env_id)


@pytest.mark.skipif(not _ADROIT_AVAILABLE, reason="external_deps/Adroit not installed")
@pytest.mark.parametrize("env_id", ADROIT_ENVS)
def test_adroit_env(env_id):
    _run_env(env_id)
