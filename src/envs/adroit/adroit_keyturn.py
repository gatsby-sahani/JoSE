import collections
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myobase.key_turn_v0 import KeyTurnEnvV0
from src.envs.adroit.adroit_base import AdroitHandMixin


class AdroitKeyTurnEnv(AdroitHandMixin, KeyTurnEnvV0):
    """KeyTurn environment using the tendon-driven Adroit hand.

    Hand geometry (adroit_keyturn.xml: pos="0 -0.3 1.45" euler="-1.5708 0 0"):
      - R_x(-90 deg): palm faces world +z, fingers point in world +y
      - Key is placed near the index/thumb fingertip region

    KeyTurnEnvV0._setup looks up finger-tip sites "IFtip" and "THtip" which
    are MyoHand-specific. Adroit uses "S_fftip" (index) and "S_thtip" (thumb).
    _setup is overridden here to use the correct Adroit site names.
    """

    def _setup(
        self,
        goal_th=3.14,
        obs_keys=KeyTurnEnvV0.DEFAULT_OBS_KEYS,
        weighted_reward_keys=KeyTurnEnvV0.DEFAULT_RWD_KEYS_AND_WEIGHTS,
        key_init_range=(0, 0),
        **kwargs,
    ):
        self.goal_th = goal_th
        self.keyhead_sid = self.sim.model.site_name2id("keyhead")
        self.IF_sid = self.sim.model.site_name2id("S_fftip")   # Adroit: index fingertip
        self.TH_sid = self.sim.model.site_name2id("S_thtip")   # Adroit: thumb tip
        self.key_init_range = key_init_range
        self.key_init_pos = self.sim.data.site_xpos[self.keyhead_sid].copy()

        BaseV0._setup(
            self,
            obs_keys=obs_keys,
            weighted_reward_keys=weighted_reward_keys,
            **kwargs,
        )
        self.init_qpos[:-1] *= 0
        # With PSJ: index 0 = PSJ, index 1 = WRJ1; both zeroed by *= 0 above.
        # Without PSJ: index 0 = WRJ1, zeroed by *= 0 above.
        # Explicit sets below are belt-and-suspenders in case a parent sets non-zero defaults.
        if self._add_wrist_pronation_supination:
            self.init_qpos[0] = 0.0  # PSJ neutral
            self.init_qpos[1] = 0.0  # WRJ1 neutral
        else:
            self.init_qpos[0] = 0.0  # WRJ1 neutral

    def get_reward_dict(self, obs_dict):
        IF_approach_dist = np.abs(
            np.linalg.norm(self.obs_dict["IFtip_approach"], axis=-1) - 0.030
        )
        TH_approach_dist = np.abs(
            np.linalg.norm(self.obs_dict["THtip_approach"], axis=-1) - 0.030
        )
        key_pos = (
            obs_dict["key_qpos"][:, :, 0]
            if obs_dict["key_qpos"].ndim == 3
            else obs_dict["key_qpos"][0]
        )
        act_mag = self._act_mag(IF_approach_dist)
        far_th = 0.1

        rwd_dict = collections.OrderedDict(
            (
                ("key_turn", key_pos),
                ("IFtip_approach", -1.0 * IF_approach_dist),
                ("THtip_approach", -1.0 * TH_approach_dist),
                ("act_reg", -1.0 * act_mag),
                ("bonus", 1.0 * (key_pos > np.pi / 2) + 1.0 * (key_pos > np.pi)),
                (
                    "penalty",
                    -1.0 * (IF_approach_dist > far_th / 2)
                    - 1.0 * (TH_approach_dist > far_th / 2),
                ),
                ("sparse", key_pos),
                ("solved", obs_dict["key_qpos"] > self.goal_th),
                ("done", (IF_approach_dist > far_th) or (TH_approach_dist > far_th)),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )
        return rwd_dict
