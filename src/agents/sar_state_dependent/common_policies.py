# import copy
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, no_type_check

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import is_image_space, maybe_transpose
from stable_baselines3.common.utils import is_vectorized_observation
from functools import partial

import distrax


class Flatten(nn.Module):
    """
    Equivalent to PyTorch nn.Flatten() layer.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x.reshape((x.shape[0], -1))


class BaseJaxPolicy(BasePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
        )

    @staticmethod
    @partial(jax.jit, static_argnames=["deterministic"])
    def sample_action(
        actor_params,
        actor_state,
        observations,
        key,
        decoder_state,
        obs_mean,
        obs_std,
        deterministic=False,
        phi=0.66,
    ):
        z_mean, z_std, n_syn = actor_state.apply_fn(actor_params, observations)

        key, subkey = jax.random.split(key)
        base_dist = distrax.MultivariateNormalDiag(loc=z_mean, scale_diag=z_std)

        tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)

        dist = distrax.Transformed(distribution=base_dist, bijector=tanh_bijector)

        z_actions, log_prob = dist.sample_and_log_prob(seed=subkey)

        syn_action = z_actions[..., :n_syn]

        no_syn_action = z_actions[..., n_syn:]

        synergies = decoder_state.apply_fn(
            decoder_state.params,
            observations,
            obs_mean,
            obs_std,
        )

        syn_action = jnp.einsum("...ij,...j->...i", synergies, syn_action)
        actor_actions = phi * syn_action + (1 - phi) * no_syn_action

        return actor_actions, log_prob

    @staticmethod
    @jax.jit
    def select_action(
        actor_state, observations, key, decoder_state, obs_mean, obs_std, phi=0.66
    ):
        key, subkey = jax.random.split(key)
        z_mean, z_std, n_syn = actor_state.apply_fn(actor_state.params, observations)

        base_dist = distrax.MultivariateNormalDiag(loc=z_mean, scale_diag=z_std)

        z_mode = base_dist.mode()

        tanh_bijector = distrax.Block(distrax.Tanh(), ndims=1)

        z_actions = tanh_bijector.forward(z_mode)

        syn_action = z_actions[..., :n_syn]
        no_syn_action = z_actions[..., n_syn:]

        synergies = decoder_state.apply_fn(
            decoder_state.params,
            observations,
            obs_mean,
            obs_std,
        )

        syn_action = jnp.einsum("...ij,...j->...i", synergies, syn_action)
        actor_actions = phi * syn_action + (1 - phi) * no_syn_action

        return actor_actions, synergies

    @no_type_check
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        obs_mean = np.array(self.vec_norm.obs_rms.mean, dtype=np.float32)
        obs_std = np.sqrt(
            np.array(
                self.vec_norm.obs_rms.var + self.vec_norm.epsilon,
                dtype=np.float32,
            )
        )

        actions = self._predict(
            observation, obs_mean, obs_std, deterministic=deterministic
        )

        # Convert to numpy, and reshape to the original action shape
        actions = np.array(actions).reshape((-1, *self.action_space.shape))

        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = np.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, (state)

    def prepare_obs(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Minimal dict support: flatten
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(
                observation[keys[0]], self.observation_space[keys[0]]
            )

            # Add batch dim and concatenate
            observation = np.concatenate(
                [
                    observation[key].reshape(-1, *self.observation_space[key].shape)
                    for key in keys
                ],
                axis=1,
            )

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            observation = np.array(observation)

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, np.ndarray)
            vectorized_env = is_vectorized_observation(
                observation, self.observation_space
            )
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, np.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode


class ContinuousCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, key) -> jnp.ndarray:

        x = Flatten()(obs)

        x = jnp.concatenate([x, action], -1)

        for n_units in self.net_arch:
            x = nn.Dense(n_units)(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=False)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    net_arch: Sequence[int]
    use_layer_norm: bool = False
    dropout_rate: Optional[float] = None
    n_critics: int = 2
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, key):
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            ContinuousCritic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True, "dropout": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            use_layer_norm=self.use_layer_norm,
            dropout_rate=self.dropout_rate,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )(obs, action, key)
        return q_values
