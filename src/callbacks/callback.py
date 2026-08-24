import os
import io
import numpy as np
import time
import imageio
import numbers
import wandb

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization


class TensorboardCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        for done in self.locals["dones"]:
            if done:
                # Log training infos
                if (
                    self.locals["log_interval"] is not None
                    and self.training_env.episode_count % self.locals["log_interval"]
                    == 0
                ):
                    self.logger.record(
                        "rollout/ep_fraction_solved",
                        safe_mean(
                            [
                                ep_info["fraction_solved"]
                                for ep_info in self.model.ep_info_buffer
                            ]
                        ),
                    )
                    self.logger.record(
                        "rollout/ep_solved",
                        safe_mean(
                            [
                                ep_info["ep_solved"]
                                for ep_info in self.model.ep_info_buffer
                            ]
                        ),
                    )
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass


class FallbackCheckpoint(CheckpointCallback):
    """
    A version of CheckpointCallback that overwrites the same file each save.
    """

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "") -> str:
        # Always overwrite the same filename (no timestep)
        filename = f"{self.name_prefix}_{checkpoint_type}fallback.{extension}"
        return os.path.join(self.save_path, filename)


class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, eval_env, verbose=0, n_eval_episodes=25):
        super().__init__(verbose)
        self._eval_freq = eval_freq
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes

    def _info_callback(self, locals_, globals_):
        # 1. Capture video frame from the first environment index at each step
        if locals_["i"] == 0:
            env = locals_["env"]
            if hasattr(env, "remotes"):
                pipe = env.remotes[0]
                pipe.send(("render", "rgb_array"))
                render = pipe.recv()
            elif hasattr(env, "envs"):
                render = env.envs[0].render("rgb_array")
            else:
                render = env.render("rgb_array")
            self._info_tracker["rollout_video"].append(render)

        # 2. Extract metrics from completed episodes across vector environments
        dones = locals_["dones"]
        infos = locals_["infos"]

        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                # Extract the episode dictionary injected by your VecMonitor
                if "episode" in info and isinstance(info["episode"], dict):
                    ep_info = info["episode"]
                    for k, v in ep_info.items():
                        # Exclude default SB3 returns (r, l) since we log mean_reward and mean_length separately
                        if k not in ["r", "l"] and isinstance(v, numbers.Number):
                            if k not in self._info_tracker:
                                self._info_tracker[k] = []
                            self._info_tracker[k].append(v)

    def _on_step(self, fps=25) -> bool:
        if self.n_calls % self._eval_freq == 0 or self.n_calls <= 1:
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self._eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way."
                    ) from e

            self._info_tracker = dict(rollout_video=[])
            start_time = time.time()

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self._eval_env,
                n_eval_episodes=self._n_eval_episodes,
                render=False,
                deterministic=True,
                return_episode_rewards=True,
                warn=True,
                callback=self._info_callback,
            )
            end_time = time.time()

            mean_reward, mean_length = (
                np.mean(episode_rewards),
                np.mean(episode_lengths),
            )
            self.logger.record("eval/time", end_time - start_time)
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_length", mean_length)

            # Log captured custom episode metrics
            for k, v in self._info_tracker.items():
                if k == "rollout_video":
                    frames = v
                    video_buffer = io.BytesIO()
                    with imageio.get_writer(
                        video_buffer, format="mp4", fps=fps
                    ) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                    video_buffer.seek(0)
                    wandb.log(
                        {
                            "eval/rollout_video": wandb.Video(
                                video_buffer, fps=fps, format="mp4"
                            )
                        },
                        step=self.num_timesteps,
                    )
                else:
                    if len(v) > 0:
                        self.logger.record(f"eval/{k}", np.mean(v))

            self.logger.dump(self.num_timesteps)
        return True
