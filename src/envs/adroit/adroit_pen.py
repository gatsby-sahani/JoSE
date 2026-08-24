import collections
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.pen_v0 import PenTwirlRandomEnvV0
from myosuite.utils.vector_math import calculate_cosine
from lattice.envs.pen import CustomPenEnv
from src.envs.adroit.adroit_base import AdroitHandMixin


class AdroitPenEnv(AdroitHandMixin, CustomPenEnv):
    """PenTwirl environment using the tendon-driven Adroit hand.

    Hand geometry (adroit_pen.xml: pos="0 -0.3 1.45" euler="-1.5708 0 0"):
      - R_x(-90 deg): palm faces world +z, fingers point in world +y
      - Pen starts in the finger region at world (0, 0.05, 1.50), axis along world x

    CustomPenEnv._setup looks up "S_grasp" which is commented out in the Adroit
    chain.xml, so _setup is overridden here to skip that lookup.
    """

    def _setup(
        self,
        obs_keys=PenTwirlRandomEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys=PenTwirlRandomEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        goal_orient_range=(-1, 1),
        enable_rsi=False,
        rsi_distance=0,
        **kwargs,
    ):
        self.target_obj_bid = self.sim.model.body_name2id("target")
        self.obj_bid = self.sim.model.body_name2id("Object")
        self.eps_ball_sid = self.sim.model.site_name2id("eps_ball")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.obj_t_sid = self.sim.model.site_name2id("object_top")
        self.obj_b_sid = self.sim.model.site_name2id("object_bottom")
        self.tar_t_sid = self.sim.model.site_name2id("target_top")
        self.tar_b_sid = self.sim.model.site_name2id("target_bottom")
        self.pen_length = np.linalg.norm(
            self.sim.model.site_pos[self.obj_t_sid]
            - self.sim.model.site_pos[self.obj_b_sid]
        )
        self.tar_length = np.linalg.norm(
            self.sim.model.site_pos[self.tar_t_sid]
            - self.sim.model.site_pos[self.tar_b_sid]
        )
        self.goal_orient_range = goal_orient_range
        self.rsi = enable_rsi
        self.rsi_distance = rsi_distance
        self.pos_align = 0
        self.rot_align = 0

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.init_qpos[:-6] *= 0
        # With PSJ: index 0 = PSJ, index 1 = WRJ1; both zeroed by *= 0 above.
        # Without PSJ: index 0 = WRJ1, zeroed by *= 0 above.
        if self._add_wrist_pronation_supination:
            self.init_qpos[0] = 0.0  # PSJ neutral
            self.init_qpos[1] = 0.0  # WRJ1 neutral
        else:
            self.init_qpos[0] = 0.0  # WRJ1 neutral

    def get_reward_dict(self, obs_dict):
        pos_err = obs_dict["obj_err_pos"]
        pos_align = np.linalg.norm(pos_err, axis=-1)
        rot_align = calculate_cosine(obs_dict["obj_rot"], obs_dict["obj_des_rot"])
        dropped = pos_align > 0.075
        act_mag = self._act_mag(pos_align)
        pos_align_diff = self.pos_align - pos_align
        rot_align_diff = rot_align - self.rot_align
        alive = ~dropped

        rwd_dict = collections.OrderedDict(
            (
                ("pos_align", -1.0 * pos_align),
                ("rot_align", rot_align),
                ("pos_align_diff", pos_align_diff),
                ("rot_align_diff", rot_align_diff),
                ("alive", alive),
                ("act_reg", -1.0 * act_mag),
                ("drop", -1.0 * dropped),
                (
                    "bonus",
                    1.0 * (rot_align > 0.9) * (pos_align < 0.075)
                    + 5.0 * (rot_align > 0.95) * (pos_align < 0.075),
                ),
                ("sparse", -1.0 * pos_align + rot_align),
                ("solved", (rot_align > 0.95) * (~dropped)),
                ("done", dropped),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = (
            np.array([0, 2]) if rwd_dict["solved"] else np.array([2, 0])
        )
        return rwd_dict
