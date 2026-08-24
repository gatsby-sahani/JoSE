from typing import Any, Callable, Dict, List, Optional, Sequence, Union, Type

import numpy as np
import jax.numpy as jnp
import jax
import flax.linen as nn
import optax
import tensorflow_probability.substrates.jax as tfp
from flax.training.train_state import TrainState
from gymnasium import spaces
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.vec_env import VecEnv
from sbx.common.policies import Flatten
from sbx.common.type_aliases import RLTrainState
from src.agents.josepi.common_policies import BaseJaxPolicy, VectorCritic
from src.agents.action_squashing_fns import ActionSquashing

tfd = tfp.distributions


class dynamics(nn.Module):
    h_dims_dynamics: List
    action_dim: int
    dynamics_observations: List
    controlled_variables: List
    drop_out_rates: List
    rank_k: int
    obs_stats: Dict[str, Union[jnp.ndarray, float]] = None
    use_play_phase_stats: bool = True

    def setup(self):

        self.dynamics = [
            nn.Sequential([nn.Dense(features=h_dim), nn.LayerNorm(), nn.relu])
            for h_dim in self.h_dims_dynamics
        ]

        self.z_dim = len(self.controlled_variables)

        if self.rank_k is None:
            output_dim_mean = self.z_dim * (self.action_dim + 1)
        else:
            output_dim_mean = (
                self.z_dim * (self.rank_k + 1) + self.rank_k * self.action_dim
            )

        output_dim_std = self.z_dim
        output_dim = output_dim_mean + output_dim_std

        self.dynamics_out = nn.Dense(features=output_dim)

        self.dropout = [
            nn.Dropout(rate=layer_i_rate) for layer_i_rate in self.drop_out_rates
        ]

    def __call__(
        self, obs, u, next_obs, curr_obs_mean, curr_obs_std, key, deterministic
    ):

        def get_log_var(x):
            """
            sigma = log(1 + exp(x))
            """
            sigma = nn.softplus(x) + 1e-6
            log_var = 2 * jnp.log(sigma)

            return log_var

        def obs_unnormalize(obs, obs_mean, obs_std):
            return obs * obs_std + obs_mean

        def obs_normalize(obs):
            return jnp.clip(
                (obs - self.obs_stats["obs_mean"])
                / jnp.sqrt(self.obs_stats["obs_var"] + self.obs_stats["obs_eps"]),
                -self.obs_stats["clip_obs"],
                self.obs_stats["clip_obs"],
            )

        # normalize obs
        if self.obs_stats is not None and self.use_play_phase_stats:
            # undo SBX normalization
            obs = obs_unnormalize(obs, curr_obs_mean, curr_obs_std)
            next_obs = obs_unnormalize(next_obs, curr_obs_mean, curr_obs_std)
            # perform normalization based on pre-train stats
            obs = obs_normalize(obs)
            next_obs = obs_normalize(next_obs)
        y_prime = next_obs[:, self.controlled_variables]

        x = obs[..., self.dynamics_observations]
        for i, fn in enumerate(self.dynamics):
            x = fn(x)
            if self.drop_out_rates[i] > 0.0:
                key, subkey = jax.random.split(key)
                x = self.dropout[i](x, deterministic, subkey)
        x = self.dynamics_out(x)

        # model the communicational channel as nonlinear in the state but affine in the action
        if self.rank_k is None:
            A_matrix, b_vector, y_prime_scale = jnp.split(
                x,
                [
                    self.z_dim * self.action_dim,
                    self.z_dim * (self.action_dim + 1),
                ],
                axis=-1,
            )
            A_matrix = A_matrix.reshape(-1, self.z_dim, self.action_dim)
            y_prime_mean = jnp.einsum("...ij,...j->...i", A_matrix, u) + b_vector

            basis_matrix = A_matrix

        else:
            # perform low-rank approximation of the linear component: A \approx B C
            B_matrix, C_matrix, b_vector, y_prime_scale = jnp.split(
                x,
                [
                    self.z_dim * self.rank_k,
                    (self.z_dim + self.action_dim) * self.rank_k,
                    (self.z_dim + self.action_dim) * self.rank_k + self.z_dim,
                ],
                axis=-1,
            )
            B_matrix = B_matrix.reshape(-1, self.z_dim, self.rank_k)
            C_matrix = C_matrix.reshape(-1, self.rank_k, self.action_dim)
            low_rank_A_matrix = jnp.einsum("...ik,...kj->...ij", B_matrix, C_matrix)
            y_prime_mean = (
                jnp.einsum("...ij,...j->...i", low_rank_A_matrix, u) + b_vector
            )

            basis_matrix = C_matrix

        y_prime_log_var = get_log_var(y_prime_scale)

        return y_prime, y_prime_mean, y_prime_log_var, basis_matrix


class synergies(nn.Module):
    action_dim: int
    z_dim: int
    controlled_variables: List
    subspace_basis_method: str

    def setup(self):

        None

    def __call__(
        self,
        obs,
        curr_obs_mean,
        curr_obs_std,
        dynamics_state,
        key,
        deterministic,
    ):

        def modified_gram_schmidt(vectors, init_i=0):
            """
            adapted from https://github.com/tensorflow/probability/blob/main/tensorflow_probability/python/math/gram_schmidt.py
            fundamental change: while_loop replaced with scan
            Args:
            vectors: A Tensor of shape `[d, n]` of `d`-dim column vectors to
              orthonormalize.

            Returns:
            A Tensor of shape `[d, n]` corresponding to the orthonormalization.
            """

            def orthogonalise_wrt_subspace(subspace_vecs, vec):
                weights = vec @ subspace_vecs
                vec = vec - subspace_vecs @ weights
                return vec

            batch_orthogonalise_wrt_subspace = jax.vmap(
                orthogonalise_wrt_subspace, in_axes=(None, 1), out_axes=(1)
            )

            def body_fn(vecs, i):
                u = jnp.nan_to_num(vecs[:, i] / jnp.linalg.norm(vecs[:, i]))
                weights = u @ vecs
                masked_weights = jnp.where(jnp.arange(num_vectors) > i, weights, 0.0)
                vecs = vecs - jnp.outer(u, masked_weights)
                return vecs, None

            num_vectors = vectors.shape[-1]
            if init_i != 0:
                null_space_basis = batch_orthogonalise_wrt_subspace(
                    vectors[:, :init_i], vectors[:, init_i:]
                )  # better to batch than scan where possible (synergies already orthonormalized)
                vectors = jnp.concatenate(
                    (vectors[:, :init_i], null_space_basis), axis=1
                )
            vectors, _ = jax.lax.scan(
                body_fn, vectors, jnp.arange(init_i, num_vectors - 1)
            )
            vec_norm = jnp.linalg.norm(vectors, axis=0, keepdims=True)
            return jnp.nan_to_num(vectors / vec_norm)

        def get_synergies(
            obs,
            curr_obs_mean,
            curr_obs_std,
            dynamics_state,
            key,
            deterministic,
        ):

            dummy_u = jnp.zeros(self.action_dim, dtype=jnp.float32)
            _, _, _, basis_matrix = dynamics_state.apply_fn(
                dynamics_state.params,
                obs[None],
                dummy_u[None],
                obs[None],
                curr_obs_mean,
                curr_obs_std,
                key,
                deterministic,
            )
            basis_matrix = basis_matrix[0]

            # (ortho)normalize rows of matrix to get synergy basis
            if self.subspace_basis_method == "normalize" or self.z_dim == 1:
                row_norms = jnp.linalg.norm(basis_matrix, axis=-1, keepdims=True)
                synergies = (basis_matrix / jnp.maximum(row_norms, 1e-8)).T
            elif self.subspace_basis_method == "gram_schmidt":
                synergies = modified_gram_schmidt(basis_matrix.T)
            elif self.subspace_basis_method == "svd":
                # can be very slow, especially for larger matrices
                U, _, Vh = jnp.linalg.svd(basis_matrix, full_matrices=False)
                synergies = (U @ Vh).T

            return synergies

        batch_get_synergies = jax.vmap(
            get_synergies, in_axes=(0, None, None, None, 0, None)
        )

        keys = jax.random.split(key, obs.shape[0])
        synergies = batch_get_synergies(
            obs,
            curr_obs_mean,
            curr_obs_std,
            dynamics_state,
            keys,
            deterministic,
        )

        return synergies


# Old checkpoints (infosyn_sac era) were serialised without this field.
# A class-level sentinel lets getattr succeed on those deserialised instances
# (instance __dict__ lookup falls through to the class attribute).
# The field is never used inside __call__, so None is safe.
synergies.controlled_variables = None  # type: ignore[attr-defined]


class Actor(nn.Module):
    net_arch: Sequence[int]
    z_dim: int
    pi_and_Q_observations: List
    action_squasher: Type[ActionSquashing]

    def setup(self):

        self.flatten = Flatten()

        policy = [
            nn.Sequential([nn.Dense(features=h_dim), nn.relu])
            for h_dim in self.net_arch
        ]
        policy.append(nn.Dense(features=self.z_dim * 2))
        self.policy = policy

    def __call__(self, obs: jnp.ndarray):

        x = self.flatten(obs[..., self.pi_and_Q_observations])
        for fn in self.policy:
            x = fn(x)

        z_mean, logit_std = jnp.split(x, 2, axis=-1)

        z_std = nn.softplus(logit_std)
        z_std = jnp.clip(z_std, min=jnp.exp(-20.0), max=jnp.exp(2.0))

        return (
            z_mean,
            z_std,
            self.action_squasher,
        )


class SACPolicy(BaseJaxPolicy):
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        controlled_variables: List,
        action_squasher: Type[ActionSquashing],
        dynamics_observations: List,
        pi_and_Q_observations: List,
        observation_space: spaces.Space,
        action_space: spaces.Box,
        lr_schedule: Schedule,
        subspace_basis_method: str,
        dynamics_dropout: float,
        dynamics_learning_rate: float,
        dynamics_lr_boundaries_and_scales: dict[int, float],
        rank_k: int,
        max_grad_norm: float,
        batch_size: int,
        pre_train_vecenv: Optional[VecEnv] = None,
        pre_train_updates: int = 0,
        use_play_phase_stats: bool = True,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        dropout_rate: float = 0.0,
        layer_norm: bool = False,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
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
        self.dropout_rate = dropout_rate
        self.layer_norm = layer_norm
        if net_arch is not None:
            if isinstance(net_arch, list):
                self.net_arch_pi = self.net_arch_qf = self.net_arch_dyn = net_arch
            else:
                self.net_arch_pi = net_arch["pi"]
                self.net_arch_qf = net_arch["qf"]
                self.net_arch_dyn = net_arch["dyn"]
        else:
            self.net_arch_pi = self.net_arch_qf = self.net_arch_dyn = [256, 256]
        self.n_critics = n_critics
        self.use_sde = use_sde
        self.activation_fn = activation_fn

        self.key = self.noise_key = jax.random.PRNGKey(0)

        self.controlled_variables = controlled_variables
        self.action_squasher = action_squasher
        self.dynamics_observations = dynamics_observations
        self.pi_and_Q_observations = pi_and_Q_observations
        self.subspace_basis_method = subspace_basis_method
        self.dynamics_dropout = dynamics_dropout
        self.dynamics_learning_rate = dynamics_learning_rate
        self.dynamics_lr_boundaries_and_scales = dynamics_lr_boundaries_and_scales
        self.pre_train_vecenv = pre_train_vecenv
        self.batch_size = batch_size
        self.rank_k = rank_k
        self.max_grad_norm = max_grad_norm
        self.pre_train_updates = pre_train_updates
        self.use_play_phase_stats = use_play_phase_stats

    def build(
        self, key: jax.Array, lr_schedule: Schedule, qf_learning_rate: float
    ) -> jax.Array:
        key, actor_key, qf_key, dropout_key, dynamics_key, synergies_key = (
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

        if self.rank_k is None:
            z_dim = len(self.controlled_variables)
        else:
            z_dim = self.rank_k

        assert (
            z_dim < action.size
        ), "The dimensionality of the synergy subspace should be smaller than the number of actuators"

        if self.pi_and_Q_observations == []:
            self.pi_and_Q_observations = list(range(self.observation_space.shape[-1]))

        self.dynamics = dynamics(
            h_dims_dynamics=self.net_arch_dyn,
            action_dim=action.size,
            dynamics_observations=self.dynamics_observations,
            controlled_variables=self.controlled_variables,
            drop_out_rates=self.dynamics_dropout,
            obs_stats=(
                {
                    "obs_mean": jnp.array(
                        self.pre_train_vecenv.obs_rms.mean, dtype=jnp.float32
                    ),
                    "obs_var": jnp.array(
                        self.pre_train_vecenv.obs_rms.var, dtype=jnp.float32
                    ),
                    "obs_eps": self.pre_train_vecenv.epsilon,
                    "clip_obs": self.pre_train_vecenv.clip_obs,
                }
                if self.pre_train_vecenv is not None
                else None
            ),
            rank_k=self.rank_k,
            use_play_phase_stats=self.use_play_phase_stats,
        )

        self.dynamics_state = TrainState.create(
            apply_fn=self.dynamics.apply,
            params=self.dynamics.init(
                dynamics_key, obs, action, obs, obs, obs, dynamics_key, True
            ),
            tx=self.optimizer_class(
                learning_rate=(
                    optax.schedules.piecewise_interpolate_schedule(
                        interpolate_type="linear",
                        init_value=self.dynamics_learning_rate,
                        boundaries_and_scales=self.dynamics_lr_boundaries_and_scales,
                    )
                    if self.dynamics_lr_boundaries_and_scales
                    else self.dynamics_learning_rate
                ),
                **self.optimizer_kwargs,
            ),
        )

        self.synergies = synergies(
            action_dim=action.size,
            z_dim=z_dim,
            subspace_basis_method=self.subspace_basis_method,
            controlled_variables=self.controlled_variables,
        )

        self.synergies_state = TrainState.create(
            apply_fn=self.synergies.apply,
            params=self.synergies.init(
                synergies_key,
                obs,
                obs,
                obs,
                self.dynamics_state,
                synergies_key,
                True,
            ),
            tx=optax.chain(
                (
                    optax.clip_by_global_norm(self.max_grad_norm)
                    if self.max_grad_norm is not None
                    else optax.identity()
                ),
                self.optimizer_class(
                    learning_rate=self.dynamics_learning_rate,  # type: ignore[call-arg] # lr_schedule(1)
                    **self.optimizer_kwargs,
                ),
            ),
        )

        self.actor = Actor(
            z_dim=z_dim,
            net_arch=self.net_arch_pi,
            pi_and_Q_observations=self.pi_and_Q_observations,
            action_squasher=self.action_squasher,
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
            pi_and_Q_observations=self.pi_and_Q_observations,
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
                self.dynamics_state,
            ),
            target_params=self.qf.init(
                {"params": qf_key, "dropout": dropout_key},
                obs,
                action,
                qf_key,
                self.dynamics_state,
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
                self.synergies_state,
                self.dynamics_state,
                obs_mean,
                obs_std,
            )
            return action, synergies
        # Trick to use gSDE: repeat sampled noise by using the same noise key
        if not self.use_sde:
            self.reset_noise()
        action, (_, synergies) = BaseJaxPolicy.sample_action(
            self.actor_state.params,
            self.actor_state,
            observation,
            self.noise_key,
            self.synergies_state,
            self.dynamics_state,
            obs_mean,
            obs_std,
        )
        return action, synergies
