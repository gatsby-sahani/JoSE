from functools import partial
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
    Callable,
)
import pathlib

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from gymnasium import spaces
from jax.typing import ArrayLike
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.vec_env import VecNormalize

from sbx.common.type_aliases import ReplayBufferSamplesNp, RLTrainState

from src.agents.josepi.policies import SACPolicy
from src.agents.josepi.common_policies import BaseJaxPolicy
from src.agents.josepi.off_policy_algorithm_jax import OffPolicyAlgorithmJax
from src.agents.action_squashing_fns import ActionSquashing


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)


class ConstantEntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> float:
        self.param("dummy_param", init_fn=lambda key: jnp.full((), self.ent_coef_init))
        return self.ent_coef_init


class SAC(OffPolicyAlgorithmJax):
    policy_aliases: ClassVar[Dict[str, Type[SACPolicy]]] = {  # type: ignore[assignment]
        "MlpPolicy": SACPolicy,
        "MultiInputPolicy": SACPolicy,
    }

    policy: SACPolicy
    action_space: spaces.Box  # type: ignore[assignment]

    def __init__(
        self,
        policy,
        env: Union[GymEnv, str],
        action_squasher: Type[ActionSquashing] = None,
        controlled_variables: List = [],
        dynamics_observations: List = [],
        pi_and_Q_observations: List = [],
        subspace_basis_method: str = "normalize",
        dynamics_dropout: float = 1e-1,
        learning_rate: Union[float, Schedule] = 3e-4,
        qf_learning_rate: Optional[float] = 3e-4,
        dynamics_learning_rate: float = 3e-4,
        dynamics_lr_boundaries: List[int] = None,
        dynamics_lr_scales: List[float] = None,
        rank_k: int = None,
        update_dyanmics_during_training: bool = True,
        use_play_phase_stats: bool = True,
        max_grad_norm: float = 0.5,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 1,
        gradient_steps: int = 1,
        policy_delay: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        ent_coef: Union[str, float] = "auto",
        target_entropy: Union[Literal["auto"], float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
        pre_train_buffer_path: Union[str, pathlib.Path] = None,
        pre_train_vecenv_path: Optional[VecNormalize] = None,
        pre_train_gradient_steps: int = 0,
        pre_train_updates: int = 0,
    ) -> None:
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            qf_learning_rate=qf_learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=(spaces.Box,),
            support_multi_env=True,
            pre_train_buffer_path=pre_train_buffer_path,
            pre_train_vecenv_path=pre_train_vecenv_path,
            pre_train_gradient_steps=pre_train_gradient_steps,
            pre_train_updates=pre_train_updates,
        )

        self.controlled_variables = controlled_variables
        self.action_squasher = action_squasher
        self.dynamics_observations = dynamics_observations
        self.pi_and_Q_observations = pi_and_Q_observations
        self.policy_delay = policy_delay
        self.ent_coef_init = ent_coef
        self.target_entropy = target_entropy
        self.subspace_basis_method = subspace_basis_method
        self.dynamics_dropout = dynamics_dropout
        self.dynamics_learning_rate = dynamics_learning_rate
        self.dynamics_lr_boundaries_and_scales = (
            dict(zip(dynamics_lr_boundaries, dynamics_lr_scales))
            if dynamics_lr_boundaries
            else None
        )
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.pre_train_updates = pre_train_updates
        self.rank_k = rank_k
        self.update_dyanmics_during_training = update_dyanmics_during_training
        self.use_play_phase_stats = use_play_phase_stats

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.controlled_variables,
                self.action_squasher,
                self.dynamics_observations,
                self.pi_and_Q_observations,
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                self.subspace_basis_method,
                self.dynamics_dropout,
                self.dynamics_learning_rate,
                self.dynamics_lr_boundaries_and_scales,
                self.rank_k,
                self.max_grad_norm,
                self.batch_size,
                self.pre_train_vecenv,
                self.pre_train_updates,
                self.use_play_phase_stats,
                **self.policy_kwargs,
            )

            if self._vec_normalize_env is not None:
                self.policy.vec_norm = self._vec_normalize_env

            assert isinstance(self.qf_learning_rate, float)

            self.key = self.policy.build(
                self.key, self.lr_schedule, self.qf_learning_rate
            )

            self.key, ent_key = jax.random.split(self.key, 2)

            self.dynamics = self.policy.dynamics  # type: ignore[assignment]
            self.synergies = self.policy.synergies  # type: ignore[assignment]
            self.actor = self.policy.actor  # type: ignore[assignment]
            self.qf = self.policy.qf  # type: ignore[assignment]

            # The entropy coefficient or entropy can be learned automatically
            # see Automating Entropy Adjustment for Maximum Entropy RL section
            # of https://arxiv.org/abs/1812.05905
            if isinstance(self.ent_coef_init, str) and self.ent_coef_init.startswith(
                "auto"
            ):

                ent_coef_init = 1.0
                if "_" in self.ent_coef_init:
                    ent_coef_init = float(self.ent_coef_init.split("_")[1])
                    assert (
                        ent_coef_init > 0.0
                    ), "The initial value of ent_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.ent_coef = EntropyCoef(ent_coef_init)
            else:
                # This will throw an error if a malformed string (different from 'auto') is passed
                assert isinstance(
                    self.ent_coef_init, float
                ), f"Entropy coef must be float when not equal to 'auto', actual: {self.ent_coef_init}"
                self.ent_coef = ConstantEntropyCoef(self.ent_coef_init)  # type: ignore[assignment]

            self.ent_coef_state = TrainState.create(
                apply_fn=self.ent_coef.apply,
                params=self.ent_coef.init(ent_key)["params"],
                tx=optax.chain(
                    (
                        optax.clip_by_global_norm(self.max_grad_norm)
                        if self.max_grad_norm is not None
                        else optax.identity()
                    ),
                    optax.adam(
                        learning_rate=(self.learning_rate),
                    ),
                ),
            )

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(
                np.float32
            )  # type: ignore
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            # self.target_entropy = float(self.target_entropy)
            self.target_entropy = np.float32(self.target_entropy)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def train(self, gradient_steps: int, batch_size: int) -> None:
        assert self.replay_buffer is not None
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.replay_buffer.sample(
            batch_size * gradient_steps * 1, env=self._vec_normalize_env
        )

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
            obs = np.concatenate(
                [data.observations[key].numpy() for key in keys], axis=1
            )
            next_obs = np.concatenate(
                [data.next_observations[key].numpy() for key in keys], axis=1
            )
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(  # type: ignore[assignment]
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.policy.dynamics_state,
            self.policy.synergies_state,
            self.key,
            (
                actor_loss_value,
                qf_loss_value,
                ent_coef_loss,
                dynamics_loss,
                entropy,
                ent_coef_value,
                next_q_values,
            ),
        ) = self._train(
            self.gamma,
            self.tau,
            self.target_entropy,
            gradient_steps,
            data,
            self.policy_delay,
            (self._n_updates + 1) % self.policy_delay,
            self.policy.qf_state,
            self.policy.actor_state,
            self.ent_coef_state,
            self.policy.dynamics_state,
            self.policy.synergies_state,
            self.key,
            self.policy.unscale_action,
            (
                jnp.array(self._vec_normalize_env.obs_rms.mean, dtype=jnp.float32)
                if self.pre_train_updates > 0
                else None
            ),
            (
                jnp.sqrt(
                    jnp.array(
                        self._vec_normalize_env.obs_rms.var
                        + self._vec_normalize_env.epsilon,
                        dtype=jnp.float32,
                    )
                )
                if self.pre_train_updates > 0
                else None
            ),
            self.update_dyanmics_during_training,
        )
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/actor_loss", actor_loss_value.item())
        self.logger.record("train/critic_loss", qf_loss_value.item())
        self.logger.record("train/dynamics_loss", dynamics_loss.item())
        self.logger.record("train/ent_coef_loss", ent_coef_loss.item())
        self.logger.record("train/ent_coef_value", ent_coef_value.item())
        self.logger.record("train/entropy", entropy.item())
        self.logger.record("train/entropy_target", self.target_entropy.item())
        self.logger.record("train/next_q_values", next_q_values.item())

    @staticmethod
    @jax.jit
    def update_dynamics(
        dynamics_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        obs_mean: jax.Array,
        obs_std: jax.Array,
        key: jax.Array,
    ):

        def dynamics_loss_fn(
            params, key, observations, actions, next_observations, obs_mean, obs_std
        ):
            def log_likelihood_diagonal_Gaussian(x, mu, log_var):
                """
                Calculate the log likelihood of x under a diagonal Gaussian distribution
                var_min is added to the variances for numerical stability
                """
                log_likelihood = -0.5 * (
                    log_var + jnp.log(2 * jnp.pi) + (x - mu) ** 2 / jnp.exp(log_var)
                )
                return log_likelihood

            y_prime, y_prime_mean, y_prime_log_var, _ = dynamics_state.apply_fn(
                params,
                observations,
                actions,
                next_observations,
                obs_mean,
                obs_std,
                key,
                deterministic=False,
            )
            dynamics_loss = -log_likelihood_diagonal_Gaussian(
                y_prime, y_prime_mean, y_prime_log_var
            ).mean(axis=-1)
            return dynamics_loss.mean()

        key, subkey = jax.random.split(key)

        dynamics_loss, grads = jax.value_and_grad(dynamics_loss_fn, has_aux=False)(
            dynamics_state.params,
            subkey,
            observations,
            actions,
            next_observations,
            obs_mean,
            obs_std,
        )
        dynamics_state = dynamics_state.apply_gradients(grads=grads)

        return dynamics_state, dynamics_loss, key

    @staticmethod
    @partial(jax.jit)
    def update_critic(
        gamma: float,
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        observations: jax.Array,
        actions: jax.Array,
        next_observations: jax.Array,
        rewards: jax.Array,
        dones: jax.Array,
        obs_mean: jax.Array,
        obs_std: jax.Array,
        key: jax.Array,
    ):
        (
            key,
            dropout_key_target,
            dropout_key_current,
            actor_key,
            critic_key,
            next_critic_key,
        ) = jax.random.split(key, 6)

        next_state_actions, (next_log_prob, _) = BaseJaxPolicy.sample_action(
            actor_state.params,
            actor_state,
            next_observations,
            actor_key,
            synergies_state,
            dynamics_state,
            obs_mean,
            obs_std,
            deterministic=False,
        )

        qf_next_values = qf_state.apply_fn(
            qf_state.target_params,
            next_observations,
            next_state_actions,
            next_critic_key,
            dynamics_state,
            rngs={"dropout": dropout_key_target},
        )

        def mse_loss(
            params: flax.core.FrozenDict,
            dynamics_state: TrainState,
            critic_key: jax.Array,
            dropout_key: jax.Array,
        ) -> jax.Array:
            # shape is (n_critics, batch_size, 1)
            current_q_values = qf_state.apply_fn(
                params,
                observations,
                actions,
                critic_key,
                dynamics_state,
                rngs={"dropout": dropout_key},
            )
            return 0.5 * ((target_q_values - current_q_values) ** 2).mean(axis=1).sum()

        ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})

        next_q_values = jnp.min(qf_next_values, axis=0)
        # td error + entropy term
        next_q_values = next_q_values - ent_coef_value * next_log_prob.reshape(-1, 1)
        # shape is (batch_size, 1)

        target_q_values = (
            rewards.reshape(-1, 1) + (1 - dones.reshape(-1, 1)) * gamma * next_q_values
        )

        qf_loss_value, grads = jax.value_and_grad(mse_loss, has_aux=False)(
            qf_state.params, dynamics_state, critic_key, dropout_key_current
        )
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (
                qf_loss_value.astype(np.float32),
                next_q_values.mean().astype(np.float32),
                ent_coef_value.astype(np.float32),
            ),
            key,
        )

    @staticmethod
    @partial(jax.jit)
    def update_actor(
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        observations: jax.Array,
        obs_mean: jax.Array,
        obs_std: jax.Array,
        key: jax.Array,
    ):
        key, dropout_key, dynamics_key, critic_key = jax.random.split(key, 4)

        def actor_loss(
            params: flax.core.FrozenDict, synergies_state, dynamics_state, key
        ) -> Tuple[jax.Array, jax.Array]:
            actor_actions, (log_prob, _) = BaseJaxPolicy.sample_action(
                params,
                actor_state,
                observations,
                key,
                synergies_state,
                dynamics_state,
                obs_mean,
                obs_std,
                deterministic=False,
            )

            qf_pi = qf_state.apply_fn(
                qf_state.params,
                observations,
                actor_actions,
                critic_key,
                dynamics_state,
                rngs={"dropout": dropout_key},
            )
            # Take min among all critics (mean for droq)
            min_qf_pi = jnp.min(qf_pi, axis=0)

            log_prob = log_prob.reshape(-1, 1)

            ent_coef_value = ent_coef_state.apply_fn({"params": ent_coef_state.params})
            actor_loss = (ent_coef_value * log_prob - min_qf_pi).mean()
            return actor_loss, -log_prob.mean()

        (
            (actor_loss_value, entropy),
            grads,
        ) = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params, synergies_state, dynamics_state, dynamics_key)
        actor_state = actor_state.apply_gradients(grads=grads)

        return (
            actor_state,
            qf_state,
            actor_loss_value,
            key,
            entropy,
        )

    @staticmethod
    @jax.jit
    def soft_update(tau: float, qf_state: RLTrainState) -> RLTrainState:
        qf_state = qf_state.replace(
            target_params=optax.incremental_update(
                qf_state.params, qf_state.target_params, tau
            )
        )
        return qf_state

    @staticmethod
    @jax.jit
    def update_temperature(
        target_entropy: ArrayLike, ent_coef_state: TrainState, entropy: float
    ):
        def temperature_loss(temp_params: flax.core.FrozenDict) -> jax.Array:
            ent_coef_value = ent_coef_state.apply_fn({"params": temp_params})
            ent_coef_loss = jnp.log(ent_coef_value) * (entropy - target_entropy).mean()  # type: ignore[union-attr]
            return ent_coef_loss, (entropy, ent_coef_value)

        (ent_coef_loss, (entropy, ent_coef_value)), grads = jax.value_and_grad(
            temperature_loss, has_aux=True
        )(ent_coef_state.params)
        ent_coef_state = ent_coef_state.apply_gradients(grads=grads)

        return ent_coef_state, ent_coef_loss, entropy, ent_coef_value

    @classmethod
    @partial(jax.jit, static_argnames=["cls"])
    def update_actor_and_temperature(
        cls,
        actor_state: TrainState,
        qf_state: RLTrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        observations: jax.Array,
        target_entropy: ArrayLike,
        obs_mean: jax.Array,
        obs_std: jax.Array,
        key: jax.Array,
    ):
        (
            actor_state,
            qf_state,
            actor_loss_value,
            key,
            entropy,
        ) = cls.update_actor(
            actor_state,
            qf_state,
            ent_coef_state,
            dynamics_state,
            synergies_state,
            observations,
            obs_mean,
            obs_std,
            key,
        )
        ent_coef_state, ent_coef_loss_value, entropy, ent_coef_value = (
            cls.update_temperature(target_entropy, ent_coef_state, entropy)
        )
        return (
            actor_state,
            qf_state,
            ent_coef_state,
            actor_loss_value.astype(jnp.float32),
            ent_coef_loss_value.astype(jnp.float32),
            entropy.astype(jnp.float32),
            ent_coef_value.astype(jnp.float32),
            key,
        )

    @classmethod
    @partial(
        jax.jit,
        static_argnames=[
            "cls",
            "gradient_steps",
            "policy_delay",
            "policy_delay_offset",
            "unscale_action",
            "update_dyanmics_during_training",
        ],
    )
    def _train(
        cls,
        gamma: float,
        tau: float,
        target_entropy: ArrayLike,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        policy_delay: int,
        policy_delay_offset: int,
        qf_state: RLTrainState,
        actor_state: TrainState,
        ent_coef_state: TrainState,
        dynamics_state: TrainState,
        synergies_state: TrainState,
        key: jax.Array,
        unscale_action: Callable,
        obs_mean: jax.Array,
        obs_std: jax.Array,
        update_dyanmics_during_training: bool,
    ):

        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // (gradient_steps * 1)

        def one_update_dynamics(i: int, carry: Dict[str, Any]) -> Dict[str, Any]:
            # Note: this method must be defined inline because
            # `fori_loop` expect a signature fn(index, carry) -> carry
            dynamics_state = carry["dynamics_state"]
            key = carry["key"]
            info = carry["info"]
            batch_obs = jax.lax.dynamic_slice_in_dim(
                data.observations, i * batch_size, batch_size
            )
            batch_scaled_act = jax.lax.dynamic_slice_in_dim(
                data.actions, i * batch_size, batch_size
            )
            batch_act = unscale_action(
                batch_scaled_act
            )  # sbx SAC scales the actions between (-1,1) before storing them
            batch_next_obs = jax.lax.dynamic_slice_in_dim(
                data.next_observations, i * batch_size, batch_size
            )

            dynamics_state, dynamics_loss, key = cls.update_dynamics(
                dynamics_state,
                batch_obs,
                batch_act,
                batch_next_obs,
                obs_mean,
                obs_std,
                key,
            )

            info = {"dynamics_loss": dynamics_loss}

            return {
                "dynamics_state": dynamics_state,
                "key": key,
                "info": info,
            }

        dynamics_carry = {
            "dynamics_state": dynamics_state,
            "key": key,
            "info": {
                "dynamics_loss": jnp.array(0.0, dtype=jnp.float32),
            },
        }

        if update_dyanmics_during_training:
            dynamics_carry = jax.lax.fori_loop(
                0, gradient_steps * 1, one_update_dynamics, dynamics_carry
            )

        carry = {
            "actor_state": actor_state,
            "qf_state": qf_state,
            "ent_coef_state": ent_coef_state,
            "dynamics_state": dynamics_carry["dynamics_state"],
            "synergies_state": synergies_state,
            "key": key,
            "info": {
                "actor_loss": jnp.array(0.0, dtype=jnp.float32),
                "qf_loss": jnp.array(0.0, dtype=jnp.float32),
                "ent_coef_loss": jnp.array(0.0, dtype=jnp.float32),
                "dynamics_loss": jnp.array(0.0, dtype=jnp.float32),
                "entropy": jnp.array(0.0, dtype=jnp.float32),
                "ent_coef_value": jnp.array(0.0, dtype=jnp.float32),
                "next_q_values": jnp.array(0.0, dtype=jnp.float32),
            },
        }

        def one_update(i: int, carry: Dict[str, Any]) -> Dict[str, Any]:
            # Note: this method must be defined inline because
            # `fori_loop` expect a signature fn(index, carry) -> carry
            actor_state = carry["actor_state"]
            qf_state = carry["qf_state"]
            ent_coef_state = carry["ent_coef_state"]
            dynamics_state = carry["dynamics_state"]
            synergies_state = carry["synergies_state"]
            key = carry["key"]
            info = carry["info"]
            batch_obs = jax.lax.dynamic_slice_in_dim(
                data.observations, i * batch_size, batch_size
            )
            batch_scaled_act = jax.lax.dynamic_slice_in_dim(
                data.actions, i * batch_size, batch_size
            )
            batch_act = unscale_action(
                batch_scaled_act
            )  # sbx SAC scales the actions between (-1,1) before storing them
            batch_next_obs = jax.lax.dynamic_slice_in_dim(
                data.next_observations, i * batch_size, batch_size
            )
            batch_rew = jax.lax.dynamic_slice_in_dim(
                data.rewards, i * batch_size, batch_size
            )
            batch_done = jax.lax.dynamic_slice_in_dim(
                data.dones, i * batch_size, batch_size
            )

            dynamics_loss = info["dynamics_loss"]

            (
                actor_state,
                qf_state,
                ent_coef_state,
                actor_loss_value,
                ent_coef_loss_value,
                entropy,
                ent_coef_value,
                key,
            ) = jax.lax.cond(
                (policy_delay_offset + i) % policy_delay == 0,
                # If True:
                cls.update_actor_and_temperature,
                # If False:
                # lambda *_: (actor_state, qf_state, ent_coef_state, info["actor_loss"], info["ent_coef_loss"], info["entropy"], info["ent_coef_value"], key),
                partial(
                    lambda actor_state, qf_state, ent_coef_state, dynamics_state, synergies_state, batch_obs, target_entropy, obs_mean, obs_std, key: (
                        actor_state,
                        qf_state,
                        ent_coef_state,
                        info["actor_loss"],
                        info["ent_coef_loss"],
                        info["entropy"],
                        info["ent_coef_value"],
                        key,
                    )
                ),
                actor_state,
                qf_state,
                ent_coef_state,
                dynamics_state,
                synergies_state,
                batch_obs,
                target_entropy,
                obs_mean,
                obs_std,
                key,
            )

            (
                qf_state,
                (qf_loss_value, next_q_values, ent_coef_value),
                key,
            ) = cls.update_critic(
                gamma,
                actor_state,
                qf_state,
                ent_coef_state,
                dynamics_state,
                synergies_state,
                batch_obs,
                batch_act,
                batch_next_obs,
                batch_rew,
                batch_done,
                obs_mean,
                obs_std,
                key,
            )
            qf_state = cls.soft_update(tau, qf_state)

            info = {
                "actor_loss": actor_loss_value,
                "qf_loss": qf_loss_value,
                "ent_coef_loss": ent_coef_loss_value,
                "dynamics_loss": dynamics_loss,
                "entropy": entropy,
                "ent_coef_value": ent_coef_value,
                "next_q_values": next_q_values,
            }

            return {
                "actor_state": actor_state,
                "qf_state": qf_state,
                "ent_coef_state": ent_coef_state,
                "dynamics_state": dynamics_state,
                "synergies_state": synergies_state,
                "key": key,
                "info": info,
            }

        update_carry = jax.lax.fori_loop(0, gradient_steps, one_update, carry)

        return (
            update_carry["qf_state"],
            update_carry["actor_state"],
            update_carry["ent_coef_state"],
            update_carry["dynamics_state"],
            update_carry["synergies_state"],
            update_carry["key"],
            (
                update_carry["info"]["actor_loss"],
                update_carry["info"]["qf_loss"],
                update_carry["info"]["ent_coef_loss"],
                dynamics_carry["info"]["dynamics_loss"],
                update_carry["info"]["entropy"],
                update_carry["info"]["ent_coef_value"],
                update_carry["info"]["next_q_values"],
            ),
        )

    def pre_train(self, batch_size: int, gradient_steps: int) -> None:
        # Sample all at once for efficiency (so we can jit the for loop)
        data = self.pre_train_replay_buffer.sample(batch_size * gradient_steps)

        if isinstance(data.observations, dict):
            keys = list(self.observation_space.keys())  # type: ignore[attr-defined]
            obs = np.concatenate(
                [data.observations[key].numpy() for key in keys], axis=1
            )
            next_obs = np.concatenate(
                [data.next_observations[key].numpy() for key in keys], axis=1
            )
        else:
            obs = data.observations.numpy()
            next_obs = data.next_observations.numpy()

        # Convert to numpy
        data = ReplayBufferSamplesNp(  # type: ignore[assignment]
            obs,
            data.actions.numpy(),
            next_obs,
            data.dones.numpy().flatten(),
            data.rewards.numpy().flatten(),
        )

        (
            self.policy.dynamics_state,
            self.key,
            dynamics_loss,
        ) = self._pre_train(
            gradient_steps,
            data,
            self.policy.dynamics_state,
            self.key,
            self.policy.unscale_action,
        )
        self.logger.record("pretrain/dynamics_loss", dynamics_loss.item())

    @classmethod
    @partial(jax.jit, static_argnames=["cls", "gradient_steps", "unscale_action"])
    def _pre_train(
        cls,
        gradient_steps: int,
        data: ReplayBufferSamplesNp,
        dynamics_state: TrainState,
        key: jax.Array,
        unscale_action: Callable,
    ):

        assert data.observations.shape[0] % gradient_steps == 0
        batch_size = data.observations.shape[0] // (gradient_steps * 1)

        obs_mean = jnp.zeros(data.observations.shape[1], dtype=jnp.float32)
        obs_std = jnp.ones(data.observations.shape[1], dtype=jnp.float32)

        def one_update_dynamics(i: int, carry: Dict[str, Any]) -> Dict[str, Any]:
            # Note: this method must be defined inline because
            # `fori_loop` expect a signature fn(index, carry) -> carry
            dynamics_state = carry["dynamics_state"]
            key = carry["key"]
            info = carry["info"]
            batch_obs = jax.lax.dynamic_slice_in_dim(
                data.observations, i * batch_size, batch_size
            )

            batch_scaled_act = jax.lax.dynamic_slice_in_dim(
                data.actions, i * batch_size, batch_size
            )
            batch_act = unscale_action(
                batch_scaled_act
            )  # sbx SAC scales the actions between (-1,1) before storing them
            batch_next_obs = jax.lax.dynamic_slice_in_dim(
                data.next_observations, i * batch_size, batch_size
            )

            dynamics_state, dynamics_loss, key = cls.update_dynamics(
                dynamics_state,
                batch_obs,
                batch_act,
                batch_next_obs,
                obs_mean,
                obs_std,
                key,
            )

            info = {"dynamics_loss": dynamics_loss}

            return {
                "dynamics_state": dynamics_state,
                "key": key,
                "info": info,
            }

        dynamics_carry = {
            "dynamics_state": dynamics_state,
            "key": key,
            "info": {
                "dynamics_loss": jnp.array(0.0, dtype=jnp.float32),
            },
        }

        dynamics_carry = jax.lax.fori_loop(
            0, gradient_steps, one_update_dynamics, dynamics_carry
        )

        return (
            dynamics_carry["dynamics_state"],
            dynamics_carry["key"],
            dynamics_carry["info"]["dynamics_loss"],
        )
