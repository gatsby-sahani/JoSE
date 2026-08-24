import collections
import numpy as np
from lattice.envs.reorient import CustomReorientEnv
from src.envs.adroit.adroit_base import AdroitHandMixin


class AdroitReorientEnv(AdroitHandMixin, CustomReorientEnv):
    """DieReorient environment using the tendon-driven Adroit hand.

    Hand geometry (adroit_die.xml: pos="0 -0.3 1.45" euler="-1.5708 0 0"):
      - R_x(-90 deg): palm faces world +z, fingers point in world +y
      - Die starts in the palm cup at world (0, 0.04, 1.52)
    """

    def get_reward_dict(self, obs_dict):
        pos_dist_new = np.abs(np.linalg.norm(self.obs_dict["pos_err"], axis=-1))
        rot_dist_new = np.abs(np.linalg.norm(self.obs_dict["rot_err"], axis=-1))
        pos_dist_diff = self.pos_dist - pos_dist_new
        rot_dist_diff = self.rot_dist - rot_dist_new
        act_mag = self._act_mag(pos_dist_new)
        drop = pos_dist_new > self.drop_th

        rwd_dict = collections.OrderedDict(
            (
                ("pos_dist", -1.0 * pos_dist_new),
                ("rot_dist", -1.0 * rot_dist_new),
                ("pos_dist_diff", pos_dist_diff),
                ("rot_dist_diff", rot_dist_diff),
                ("alive", ~drop),
                ("bonus", 1.0 * (pos_dist_new < 2 * self.pos_th) + 1.0 * (pos_dist_new < self.pos_th)),
                ("penalty", -1.0 * drop),
                ("act_reg", -1.0 * act_mag),
                ("sparse", -rot_dist_new - 10.0 * pos_dist_new),
                (
                    "solved",
                    (
                        (pos_dist_new < self.pos_th)
                        and (rot_dist_new < self.rot_th)
                        and (not drop)
                    )
                    * np.ones((1, 1)),
                ),
                ("done", drop),
            )
        )
        rwd_dict["dense"] = np.sum(
            [wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0
        )

        self.sim.model.site_rgba[self.success_indicator_sid, :2] = (
            np.array([0, 2]) if rwd_dict["solved"] else np.array([2, 0])
        )
        return rwd_dict
