import collections
import numpy as np
from lattice.envs.baoding import CustomBaodingEnv
from src.envs.adroit.adroit_base import AdroitHandMixin


class AdroitBaodingEnv(AdroitHandMixin, CustomBaodingEnv):
    """BaodingBalls environment using the tendon-driven Adroit hand.

    Hand geometry (adroit_baoding.xml: pos="0 -0.3 1.45" euler="-1.5708 0 0"):
      - R_x(-90 deg): local -y (palm inner surface) → world +z (palm faces up)
      - Palm centre at world ≈ (0, -0.01, 1.44); fingers point in world +y
      - Balls rest at world z ≈ 1.47, orbit centre ≈ (0.011, 0.05)
    """

    def _setup(self, drop_th=1.25, center_pos=None, **kwargs):
        super()._setup(drop_th=drop_th, **kwargs)
        self.center_pos = center_pos if center_pos is not None else [0.011, 0.05]

    def get_obs_dict(self, sim):
        obs_dict = {}
        obs_dict["time"] = np.array([sim.data.time])
        obs_dict["hand_pos"] = sim.data.qpos[:-14].copy()
        obs_dict["object1_pos"] = sim.data.site_xpos[self.object1_sid].copy()
        obs_dict["object2_pos"] = sim.data.site_xpos[self.object2_sid].copy()
        obs_dict["object1_velp"] = sim.data.qvel[-12:-9].copy() * self.dt
        obs_dict["object2_velp"] = sim.data.qvel[-6:-3].copy() * self.dt
        obs_dict["target1_pos"] = sim.data.site_xpos[self.target1_sid].copy()
        obs_dict["target2_pos"] = sim.data.site_xpos[self.target2_sid].copy()
        obs_dict["target1_err"] = obs_dict["target1_pos"] - obs_dict["object1_pos"]
        obs_dict["target2_err"] = obs_dict["target2_pos"] - obs_dict["object2_pos"]
        obs_dict["act"] = np.zeros(sim.model.nu)
        return obs_dict

    def get_reward_dict(self, obs_dict):
        target1_dist = np.linalg.norm(obs_dict["target1_err"], axis=-1)
        target2_dist = np.linalg.norm(obs_dict["target2_err"], axis=-1)
        target_dist = target1_dist + target2_dist
        act_mag = self._act_mag(target1_dist)

        object1_pos = (
            obs_dict["object1_pos"][:, :, 2]
            if obs_dict["object1_pos"].ndim == 3
            else obs_dict["object1_pos"][2]
        )
        object2_pos = (
            obs_dict["object2_pos"][:, :, 2]
            if obs_dict["object2_pos"].ndim == 3
            else obs_dict["object2_pos"][2]
        )
        is_fall_1 = object1_pos < self.drop_th
        is_fall_2 = object2_pos < self.drop_th
        is_fall = np.logical_or(is_fall_1, is_fall_2)

        rwd_dict = collections.OrderedDict(
            (
                ("pos_dist_1", -1.0 * target1_dist),
                ("pos_dist_2", -1.0 * target2_dist),
                ("alive", ~is_fall),
                ("act_reg", -1.0 * act_mag),
                ("sparse", -target_dist),
                (
                    "solved",
                    (target1_dist < self.proximity_th)
                    * (target2_dist < self.proximity_th)
                    * (~is_fall),
                ),
                ("done", is_fall),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        self.sim.model.geom_rgba[self.object1_gid, :2] = (
            np.array([1, 1])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )
        self.sim.model.geom_rgba[self.object2_gid, :2] = (
            np.array([0.9, 0.7])
            if target1_dist < self.proximity_th
            else np.array([0.5, 0.5])
        )

        return rwd_dict
