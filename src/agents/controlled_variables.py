from myosuite.utils import gym
import numpy as np


Q_TO_CONTROL_MYOHAND = [
    "pro_sup",
    "deviation",
    "flexion",
    "cmc_abduction",
    "cmc_flexion",
    "mp_flexion",
    "ip_flexion",
    "mcp2_flexion",
    "mcp2_abduction",
    "pm2_flexion",
    "md2_flexion",
    "mcp3_flexion",
    "mcp3_abduction",
    "pm3_flexion",
    "md3_flexion",
    "mcp4_flexion",
    "mcp4_abduction",
    "pm4_flexion",
    "md4_flexion",
    "mcp5_flexion",
    "mcp5_abduction",
    "pm5_flexion",
    "md5_flexion",
]

Q_TO_CONTROL_ADROIT = [
    "WRJ1", "WRJ0",
    "FFJ3", "FFJ2", "FFJ1", "FFJ0",
    "MFJ3", "MFJ2", "MFJ1", "MFJ0",
    "RFJ3", "RFJ2", "RFJ1", "RFJ0",
    "LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0",
    "THJ4", "THJ3", "THJ2", "THJ1", "THJ0",
]

_ADROIT_ENV_IDS = {
    "CustomAdroitBaodingP1-v1",
    "CustomAdroitDieReorientP2-v0",
    "CustomAdroitKeyTurnRandom-v0",
    "CustomAdroitPenTwirlRandom-v0",
}

# Keep old name as alias for backwards compatibility
Q_TO_CONTROL = Q_TO_CONTROL_MYOHAND


class ObsWrapperQvel(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        super().__init__(env)

        if cfg.env.env_id in _ADROIT_ENV_IDS:
            add_psj = cfg.env.env_kwargs.get("add_wrist_pronation_supination", False)
            q_to_control = (["PSJ"] if add_psj else []) + Q_TO_CONTROL_ADROIT
        else:
            q_to_control = Q_TO_CONTROL_MYOHAND
        self.hand_joint_ids = [
            self.env.unwrapped.sim.model.joint_name2id(q) for q in q_to_control
        ]

        self.num_pi_and_Q_observations = self.env.observation_space.shape[0]
        self.num_controlled_variables = len(q_to_control)
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.num_pi_and_Q_observations + self.num_controlled_variables,),
            dtype=self.env.observation_space.dtype,
        )

        self.pi_and_Q_observations = list(range(self.num_pi_and_Q_observations))
        self.dynamics_observations = self.pi_and_Q_observations.copy()
        task_variables_keys = cfg.env.task_variables_keys
        task_variables = [
            idx
            for key in task_variables_keys
            for idx in list(self.env.get_wrapper_attr("key_idx")[key])
        ]
        for task_variable in task_variables:
            self.dynamics_observations.remove(task_variable)

        self.controlled_variables = [
            self.num_pi_and_Q_observations + i
            for i in range(self.num_controlled_variables)
        ]

    def observation(self, obs):
        return np.concatenate(
            (obs, self.env.unwrapped.sim.data.qvel[self.hand_joint_ids]), axis=0
        )


def get_controlled_variables(cfg):
    if cfg.agent.name in [
        "josepi",
        "play_phase_agent",
        "sac_sar_state_independent",
        "sac_sar_state_dependent",
    ]:
        return ObsWrapperQvel

    return None
