import pathlib
import warnings

import numpy as np
from importlib_resources import files

# myosuite registers some environments twice (myobase + myoedits), and the
# lattice package registers CustomMyoChallenge envs with stale config before
# we override them below — suppress the resulting gymnasium warnings.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
    from myosuite.utils import gym
    import lattice.envs.baoding
    import lattice.envs.reorient

import src.envs.adroit.adroit_baoding
import src.envs.adroit.adroit_reorient
import src.envs.adroit.adroit_keyturn
import src.envs.adroit.adroit_pen
from src.envs.adroit.adroit_utils import build_adroit_assets_include

build_adroit_assets_include()

_ASSETS_DIR = pathlib.Path(__file__).parent / "adroit" / "assets"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.envs.registration")
    gym.envs.registration.register(
        id="CustomMyoChallengeBaodingP1-v1",
        entry_point="lattice.envs.baoding:CustomBaodingEnv",
        max_episode_steps=200,
        kwargs={
            "model_path": str(
                files("myosuite").joinpath("envs/myo/assets/hand/myohand_baoding.xml")
            ),
            "normalize_act": True,
            "goal_xrange": (0.025, 0.025),
            "goal_yrange": (0.028, 0.028),
        },
    )

    gym.envs.registration.register(
        id="CustomMyoChallengeDieReorientP2-v0",
        entry_point="lattice.envs.reorient:CustomReorientEnv",
        max_episode_steps=150,
        kwargs={
            "model_path": str(
                files("myosuite").joinpath("envs/myo/assets/hand/myohand_die.xml")
            ),
            "normalize_act": True,
            "frame_skip": 5,
            "goal_pos": (-0.020, 0.020),
            "goal_rot": (-3.14, 3.14),
            "obj_size_change": 0.007,
            "obj_friction_change": (0.2, 0.001, 0.00002),
        },
    )

gym.envs.registration.register(
    id="CustomAdroitBaodingP1-v1",
    entry_point="src.envs.adroit.adroit_baoding:AdroitBaodingEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_ASSETS_DIR / "adroit_baoding.xml"),
        "normalize_act": True,
        "goal_xrange": (0.025, 0.025),
        "goal_yrange": (0.028, 0.028),
    },
)

gym.envs.registration.register(
    id="CustomAdroitDieReorientP2-v0",
    entry_point="src.envs.adroit.adroit_reorient:AdroitReorientEnv",
    max_episode_steps=150,
    kwargs={
        "model_path": str(_ASSETS_DIR / "adroit_die.xml"),
        "normalize_act": True,
        "frame_skip": 5,
        "goal_pos": (-0.020, 0.020),
        "goal_rot": (-3.14, 3.14),
        "obj_size_change": 0.007,
        "obj_friction_change": (0.2, 0.001, 0.00002),
    },
)

gym.envs.registration.register(
    id="CustomAdroitKeyTurnRandom-v0",
    entry_point="src.envs.adroit.adroit_keyturn:AdroitKeyTurnEnv",
    max_episode_steps=200,
    kwargs={
        "model_path": str(_ASSETS_DIR / "adroit_keyturn.xml"),
        "normalize_act": True,
        "key_init_range": (-np.pi / 2, np.pi / 2),
        "goal_th": 2 * np.pi,
    },
)

gym.envs.registration.register(
    id="CustomAdroitPenTwirlRandom-v0",
    entry_point="src.envs.adroit.adroit_pen:AdroitPenEnv",
    max_episode_steps=50,
    kwargs={
        "model_path": str(_ASSETS_DIR / "adroit_pen.xml"),
        "normalize_act": True,
        "frame_skip": 5,
    },
)
