import numpy as np
import mujoco
from myosuite.utils import gym
from gymnasium import spaces
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class ActInfoWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, done, terminated, info = self.env.step(action)
        info["act"] = self.env.sim.data.act.copy()
        return obs, reward, done, terminated, info


class RenderWrapper(gym.Wrapper):
    render_mode = "rgb_array"

    def __init__(self, env, cfg):
        super().__init__(env)

        self.camera = mujoco.MjvCamera()
        if cfg.env.camera is not None:
            if cfg.env.camera.type == "tracking":
                self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.camera.trackbodyid = self.env.unwrapped.sim.model.name2id(
                    cfg.env.camera.tracked_body, "body"
                )
            else:
                self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.camera.lookat = cfg.env.camera.lookat
            self.camera.distance = cfg.env.camera.distance
            self.camera.azimuth = cfg.env.camera.azimuth
            self.camera.elevation = cfg.env.camera.elevation
        self.cfg = cfg

    def render(self):
        if self.cfg.env.camera is not None:
            return self.env.unwrapped.sim.renderer.render_offscreen(
                width=240, height=320, camera_id=self.camera
            )
        else:
            return self.env.unwrapped.sim.renderer.render_offscreen(
                width=240, height=320
            )


class SparseRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = info["solved"] * 1.0  # pose_dist<self.pose_thd
        return obs, reward, terminated, truncated, info


# modified from https://github.com/HumanCompatibleAI/imitation/blob/a8b079c469bb145d1954814f22488adff944aa0d/src/imitation/policies/replay_buffer_wrapper.py
class ReplayBufferWrapper(ReplayBuffer):
    """Store more transitions in the buffer than are used for learning."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        # *,
        # device: Union[th.device, str] = "auto",
        # n_envs: int = 1,
        # optimize_memory_usage: bool = False,
        **kwargs,
    ):
        """

        Args:
            buffer_size: Max number of elements in the buffer
            observation_space: Observation space
            action_space: Action space
            **kwargs: keyword arguments for ReplayBuffer.
        """
        assert not isinstance(observation_space, spaces.Dict)

        self.total_buffer_size = kwargs["total_buffer_size"]
        self.total_buffer_size_per_env = max(
            kwargs["total_buffer_size"] // kwargs["n_envs"], 1
        )
        self.buffer_size_per_env = max(buffer_size // kwargs["n_envs"], 1)
        del kwargs["total_buffer_size"]

        self.replay_buffer = ReplayBuffer(
            self.total_buffer_size,  # buffer_size,
            observation_space,
            action_space,
            **kwargs,
        )
        _base_kwargs = {k: v for k, v in kwargs.items() if k in ["device", "n_envs"]}
        # super().__init__(buffer_size, observation_space, action_space, **_base_kwargs)

    @property
    def pos(self) -> int:
        return self.replay_buffer.pos

    @pos.setter
    def pos(self, pos: int):
        self.replay_buffer.pos = pos

    @property
    def full(self) -> bool:
        return self.replay_buffer.full

    @full.setter
    def full(self, full: bool):
        self.replay_buffer.full = full

    def sample(self, *args, **kwargs):

        batch_size = args[0]
        env = kwargs["env"]

        upper_bound = self.total_buffer_size_per_env if self.full else self.pos
        lower_bound = max(upper_bound - self.buffer_size_per_env, 0)
        batch_inds = np.random.randint(lower_bound, upper_bound, size=batch_size)
        samples = self.replay_buffer._get_samples(batch_inds, env)

        return ReplayBufferSamples(
            samples.observations,
            samples.actions,
            samples.next_observations,
            samples.dones,
            samples.rewards,
        )

    def add(self, *args, **kwargs):
        self.replay_buffer.add(*args, **kwargs)
        return None

    def _get_samples(self):
        raise NotImplementedError(
            "_get_samples() is intentionally not implemented."
            "This method should not be called.",
        )
