from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from sklearn.preprocessing import StandardScaler

from sbx.common.policies import Flatten
from src.agents.sar_state_dependent.common_policies import BaseJaxPolicy, VectorCritic
from sbx.common.type_aliases import RLTrainState

tfd = tfp.distributions


class decoder(nn.Module):
    h_dims_decoder: List
    action_dim: int
    n_syn: int
    obs_stats: Dict[str, Union[jnp.ndarray, float]] = None
    use_play_phase_stats: bool = True

    def setup(self):

        self.decoder = [
            nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu])
            for h_dim in self.h_dims_decoder
        ]

        self.decoder_out = nn.Dense(features=self.action_dim * self.n_syn)

    def __call__(self, obs, curr_obs_mean, curr_obs_std):

        def obs_unnormalize(obs, obs_mean, obs_std):
            return obs * obs_std + obs_mean

        def obs_normalize(obs):
            return (obs - self.obs_stats["mean"]) / self.obs_stats["scale"]

        if self.obs_stats is not None and self.use_play_phase_stats:
            # undo SBX normalization
            obs = obs_unnormalize(obs, curr_obs_mean, curr_obs_std)
            # perform normalization based on pre-train stats
            obs = obs_normalize(obs)

        x = obs

        for i, fn in enumerate(self.decoder):
            x = fn(x)
        V_matrix = self.decoder_out(x)

        V_matrix = V_matrix.reshape(-1, self.action_dim, self.n_syn)

        def modified_gram_schmidt(vectors):
            num_vectors = vectors.shape[-1]
            eps = 1e-8

            def body_fn(vecs, i):
                v = vecs[:, i]
                v_norm = jnp.linalg.norm(v) + eps
                u = jnp.nan_to_num(v / v_norm)

                vecs = vecs.at[:, i].set(u)

                weights = u @ vecs
                mask = jnp.arange(num_vectors) > i
                vecs = vecs - jnp.outer(u, jnp.where(mask, weights, 0.0))

                return vecs, None

            # Scan over the full range
            vectors, _ = jax.lax.scan(body_fn, vectors, jnp.arange(num_vectors))
            return vectors  # No final normalization needed if done in loop

        batch_modified_gram_schmidt = jax.vmap(modified_gram_schmidt)

        decoder_matrix = batch_modified_gram_schmidt(V_matrix)

        return decoder_matrix


class Actor(nn.Module):
    net_arch: Sequence[int]
    action_dim: int
    n_syn: int
    std_max: float = 1.0
    log_std_init: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu

    def setup(self):

        self.flatten = Flatten()

        policy = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.net_arch
        ]
        policy.append(nn.Dense(features=(self.action_dim + self.n_syn) * 2))
        self.policy = policy

    def __call__(self, obs: jnp.ndarray):

        x = self.flatten(obs)
        for fn in self.policy:
            x = fn(x)

        z_mean, logit_std = jnp.split(x, 2, axis=-1)

        z_std = nn.softplus(logit_std + self.log_std_init)
        z_std = jnp.clip(z_std, min=jnp.exp(-20.0), max=jnp.exp(2.0))

        return (
            z_mean,
            z_std,
            self.n_syn,
        )


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        std_max: float,
        state_dependent_std: bool,
        decoder_learning_rate: float,
        max_grad_norm: float,
        batch_size: int,
        n_syn: int,
        pretrain_scaler: Optional[StandardScaler] = None,
        use_play_phase_stats: bool = True,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        dropout_rate: float = 0.0,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        log_std_init: float = 0.0,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = False,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=False,
        )
        self.layer_norm = layer_norm
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = self.net_arch_dec = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
                self.net_arch_dec = net_arch["dec"]
        else:
            self.net_arch_pi = self.net_arch_qf = self.net_arch_dec = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate

        self.key = self.noise_key = jax.random.PRNGKey(0)

        self.std_max = std_max
        self.state_dependent_std = state_dependent_std
        self.log_std_init = log_std_init
        self.pretrain_scaler = pretrain_scaler
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.use_play_phase_stats = use_play_phase_stats
        self.decoder_learning_rate = decoder_learning_rate
        self.n_syn = n_syn

    def build(
        self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float
    ) -> jax.Array:
        key, actor_key, qf_key, dropout_key, decoder_key, synergies_key = (
            jax.random.split(key, 6)
        )
        key, self.key = jax.random.split(key, 2)
        self.reset_noise()

        if isinstance(self.observation_space, spaces.Dict):
            obs = jnp.array(
                [
                    spaces.flatten(
                        self.observation_space, self.observation_space.sample()
                    )
                ]
            )
        else:
            obs = jnp.array([self.observation_space.sample()])
        action = jnp.array([self.action_space.sample()])

        self.decoder = decoder(
            h_dims_decoder=self.net_arch_dec,
            action_dim=action.size,
            n_syn=self.n_syn,
            obs_stats=(
                {
                    "mean": jnp.array(self.pretrain_scaler.mean_, dtype=jnp.float32),
                    "scale": jnp.array(self.pretrain_scaler.scale_, dtype=jnp.float32),
                }
            ),
            use_play_phase_stats=self.use_play_phase_stats,
        )

        self.decoder_state = TrainState.create(
            apply_fn=self.decoder.apply,
            params=self.decoder.init(decoder_key, obs, obs, obs),
            tx=self.optimizer_class(
                learning_rate=self.decoder_learning_rate,
                **self.optimizer_kwargs,
            ),
        )

        self.actor = Actor(
            action_dim=action.size,
            n_syn=self.n_syn,
            net_arch=self.net_arch_pi,
            activation_fn=self.activation_fn,
            std_max=self.std_max,
            log_std_init=self.log_std_init,
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                (
                    optax.clip_by_global_norm(self.max_grad_norm)
                    if self.max_grad_norm is not None
                    else optax.identity()
                ),
                self.optimizer_class(
                    learning_rate=(lr_schedule(1)),
                    **self.optimizer_kwargs,
                ),
            ),
        )

        self.qf = VectorCritic(
            dropout_rate=self.dropout_rate,
            use_layer_norm=self.layer_norm,
            net_arch=self.net_arch_qf,
            n_critics=self.n_critics,
            activation_fn=self.activation_fn,
        )

        self.qf_state = RLTrainState.create(
            apply_fn=self.qf.apply,
            params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                qf_key,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                qf_key,
            ),
            tx=optax.chain(
                (
                    optax.clip_by_global_norm(self.max_grad_norm)
                    if self.max_grad_norm is not None
                    else optax.identity()
                ),
                self.optimizer_class(
                    learning_rate=(qf_learning_rate),
                    **self.optimizer_kwargs,
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.qf.apply = jax.jit(  # type: ignore[method-assign]
            self.qf.apply,
            static_argnames=("dropout_rate", "use_layer_norm"),
        )

        return key

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self._predict(obs, deterministic=deterministic)

    def _predict(
        self,
        observation: np.ndarray,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:  # type: ignore[override]
        if deterministic:
            action, synergies = BaseJaxPolicy.select_action(
                self.actor_state,
                observation,
                self.noise_key,
                self.decoder_state,
                obs_mean,
                obs_std,
            )
            return action
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        action, log_prob = BaseJaxPolicy.sample_action(
            self.actor_state.params,
            self.actor_state,
            observation,
            self.noise_key,
            self.decoder_state,
            obs_mean,
            obs_std,
        )
        return action
