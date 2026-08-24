import os
import numpy as np

from myosuite.utils import gym

from src.agents.policy_factory import load_trained_policy

from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import gc
import torch
import pickle


def get_activations(env, vec, model, episodes=2_000, percentile=80):
    """
    Vectorized version using SubprocVecEnv.

    Assumes:
        - env is a VecEnv (e.g. SubprocVecEnv)
        - each env returns info["act"] = muscle activations
    """

    n_envs = env.num_envs

    # ----------------------------
    # 1. Preview phase (parallel)
    # ----------------------------
    preview_rewards = []

    obs = env.reset()
    ep_rewards = np.zeros(n_envs)

    while len(preview_rewards) < 100:
        obs = vec.normalize_obs(obs)
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)

        ep_rewards += rewards

        for i in range(n_envs):
            if dones[i]:
                preview_rewards.append(ep_rewards[i])
                ep_rewards[i] = 0.0

                if len(preview_rewards) >= 100:
                    break

    reward_threshold = np.percentile(preview_rewards, percentile)

    # ----------------------------
    # 2. Main rollout phase
    # ----------------------------
    solved_acts = []
    solved_obs = []

    obs = env.reset()
    ep_rewards = np.zeros(n_envs)
    ep_acts = [[] for _ in range(n_envs)]
    ep_obs = [[] for _ in range(n_envs)]
    completed_episodes = 0

    pbar = tqdm(total=episodes)

    while completed_episodes < episodes:
        obs = vec.normalize_obs(obs)
        actions, _ = model.predict(obs, deterministic=False)
        obs, rewards, dones, infos = env.step(actions)

        for i in range(n_envs):
            ep_rewards[i] += rewards[i]

            # collect activations from info
            ep_acts[i].append(infos[i]["act"])

            ep_obs[i].append(obs[i])

            if dones[i]:
                if ep_rewards[i] > reward_threshold:
                    solved_acts.extend(ep_acts[i])
                    solved_obs.extend(ep_obs[i])

                ep_rewards[i] = 0.0
                ep_acts[i] = []
                ep_obs[i] = []

                completed_episodes += 1
                pbar.update(1)

                if completed_episodes >= episodes:
                    break

    pbar.close()

    return np.array(solved_acts), np.array(solved_obs)


def find_synergies(acts):
    """
    Computed % variance explained in the original muscle activation data with N synergies.

    acts: np.array; rollout data containing the muscle activations
    """
    syn_dict = {}
    for i in range(acts.shape[1]):
        pca = PCA(n_components=i + 1)
        _ = pca.fit_transform(acts)
        syn_dict[i + 1] = round(sum(pca.explained_variance_ratio_), 4)
        print("synergy #:", i + 1, "VAF:", syn_dict[i + 1])

    return syn_dict


def get_state_independent_sar(acts, n_syn=20):
    """
    Takes muscle activation data and desired n_syn as input and returns the ICA, PCA, and Scaler objects

    acts: np.array; rollout data containing the muscle activations
    n_syn: int; number of synergies to use
    """
    _ = find_synergies(acts)

    pca = PCA(n_components=n_syn)
    pca_act = pca.fit_transform(acts)

    ica = FastICA()
    pcaica_act = ica.fit_transform(pca_act)

    normalizer = MinMaxScaler((-1, 1))
    normalizer.fit(pcaica_act)

    print("A state-independent SAR has been computed using ICA-PCA.")

    return (ica, pca, normalizer)


def get_env_and_model(cfg):

    from src.envs.env_factory import make_sar_env, make_vectorized_env

    env = make_vectorized_env(
        cfg,
        make_sar_env,
        num_envs=cfg.trainer.num_training_envs,
    )

    model, vec = load_trained_policy(cfg, env, generate_sar=True)

    return env, vec, model


def get_sar_data(cfg):

    env, vec, model = get_env_and_model(cfg)

    acts, obs = get_activations(env, vec, model)

    if cfg.agent.controlled_variable in ["Qvel"]:
        n_joints = len(env.unwrapped.get_attr("hand_joint_ids")[0])
        obs = obs[:, :-n_joints]  # remove appended controlled variable from obs
    else:
        raise AssertionError("Invalid controlled variable specified.")

    env.close()
    vec.close()

    model.policy.to("cpu")

    del model, env, vec
    gc.collect()
    torch.cuda.empty_cache()

    play_data_dict = {"acts": acts, "obs": obs}

    run_path = os.path.join("play_phase_data")

    wand_run_id = cfg.agent.load_dir.split("/")[-1]

    os.makedirs(run_path, exist_ok=True)

    with open(os.path.join(run_path, f"{wand_run_id}.pkl"), "wb") as f:
        pickle.dump(play_data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return acts


class SynNoSynWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the combination of a task-general synergy space and a
    task-specific orginal space, and uses this mix to step the environment in the original action space.
    """

    def __init__(self, env, ica_pca, phi=0.66):
        super().__init__(env)
        self.ica = ica_pca[0]
        self.pca = ica_pca[1]
        self.scaler = ica_pca[2]
        self.weight = phi

        self.syn_act_space = self.pca.components_.shape[0]
        self.no_syn_act_space = env.action_space.shape[0]
        self.full_act_space = self.syn_act_space + self.no_syn_act_space

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.full_act_space,), dtype=np.float32
        )

    def action(self, act):
        syn_action = act[: self.syn_act_space]
        no_syn_action = act[self.syn_act_space :]

        syn_action = self.pca.inverse_transform(
            self.ica.inverse_transform(self.scaler.inverse_transform([syn_action]))
        )[0]
        final_action = self.weight * syn_action + (1 - self.weight) * no_syn_action

        return final_action


class SynergyWrapper(gym.ActionWrapper):
    """
    gym.ActionWrapper that reformulates the action space as the synergy space and inverse transforms
    synergy-exploiting actions back into the original muscle activation space.
    """

    def __init__(self, env, ica_pca):
        super().__init__(env)
        self.ica = ica_pca[0]
        self.pca = ica_pca[1]
        self.scaler = ica_pca[2]

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.pca.components_.shape[0],), dtype=np.float32
        )

    def action(self, act):
        action = self.pca.inverse_transform(
            self.ica.inverse_transform(self.scaler.inverse_transform([act]))
        )
        return action[0]
